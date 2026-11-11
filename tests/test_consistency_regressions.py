from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from modelctl import lifecycle, ops, receipt, runner
from modelctl.manifest import ModelManifest, StartConfig
from modelctl.system import ProcessIdentity, capture_process_identity, terminate_process_identity, reap_retained_popens, command_cwd_fingerprint


class ConsistencyRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.manifest = self.make_manifest('current')
        self.identity = ProcessIdentity(654321, 654321, 'test-birth')
        self.lock_patch = patch.object(lifecycle, 'default_lock_root', return_value=self.root / 'locks')
        self.lock_patch.start()
        self.addCleanup(self.lock_patch.stop)

    def make_manifest(self, name, cwd='.'):
        return ModelManifest(path=self.root / (name + '.toml'), id=name, model_id='test-model',
            endpoint='http://127.0.0.1:65431/v1', start=StartConfig(
                command=[sys.executable, '-c', 'import time; time.sleep(60)'], cwd=cwd,
                pid_path=name + '.pid.json', log_path=name + '.log', env={'CONSISTENCY_TEST': 'value'}))

    def active(self, manifest=None, identity=None):
        m = manifest or self.manifest
        runner.write_pid_state(m, runner._active_state(m, identity or self.identity, self.root / m.start.log_path))

    def child_cleanup(self, proc, identity):
        if capture_process_identity(proc.pid) == identity:
            terminate_process_identity(identity, timeout_sec=5)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if capture_process_identity(proc.pid) == identity:
                proc.kill()
            proc.wait(timeout=5)
        reap_retained_popens()

    def capture_spawn(self, captured):
        owner = self
        class CapturedPopen(subprocess.Popen):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                identity = capture_process_identity(self.pid)
                if identity is None:
                    self.kill()
                    self.wait()
                    raise RuntimeError('disposable process identity unavailable')
                owner.addCleanup(owner.child_cleanup, self, identity)
                captured.append((self, identity, args, kwargs))
        return CapturedPopen

    def test_doctor_relative_dead_cleanup_from_another_directory(self):
        self.active()
        path = self.root / self.manifest.start.pid_path
        with patch.object(runner, 'capture_process_identity', return_value=None), patch.object(runner, 'live_process_group_members', return_value=[]), patch.object(ops, 'doctor', return_value={'ok': True}):
            try:
                result = ops.doctor_fix(self.manifest)
            except FileNotFoundError:
                result = {'fixes': []}
        self.assertFalse(path.exists(), result)
        self.assertIn('stale_pid_state_removed', [x['code'] for x in result['fixes']])

    def test_doctor_relative_dead_and_preserved_statuses(self):
        m = self.manifest
        m.start.pid_path = str(self.root / m.start.pid_path)
        path = Path(m.start.pid_path)
        self.active()
        with patch.object(runner, 'capture_process_identity', return_value=None), patch.object(runner, 'live_process_group_members', return_value=[]), patch.object(ops, 'doctor', return_value={'ok': True}):
            result = ops.doctor_fix(m)
        self.assertFalse(path.exists())
        self.assertIn('stale_pid_state_removed', [x['code'] for x in result['fixes']])
        cases = ('group_live', 'unproven', 'pending', 'blocked', 'owner_mismatch', 'identity_mismatch')
        for case in cases:
            with self.subTest(case=case):
                self.active()
                if case == 'pending':
                    runner.write_pid_state(m, runner._pending_state(m, 'transaction'))
                elif case == 'blocked':
                    path.write_text('invalid')
                elif case == 'owner_mismatch':
                    state = runner.read_pid_state(m); state['manifest'] = '/foreign/manifest'; runner.write_pid_state(m, state)
                captures = {'group_live': None, 'unproven': OSError('inventory unavailable'), 'identity_mismatch': ProcessIdentity(654321, 654321, 'replacement')}
                capture = captures.get(case, None)
                with patch.object(runner, 'capture_process_identity', side_effect=capture if isinstance(capture, Exception) else None, return_value=None if isinstance(capture, Exception) else capture), patch.object(runner, 'live_process_group_members', return_value=[654322] if case == 'group_live' else []), patch.object(ops, 'doctor', return_value={'ok': True}):
                    result = ops.doctor_fix(m)
                self.assertTrue(path.exists(), case)
                self.assertTrue(any('preserved' in x['code'] for x in result['fixes']), result)
        self.active()
        path.chmod(0o644)
        with patch.object(ops, 'doctor', return_value={'ok': True}):
            result = ops.doctor_fix(m)
        self.assertTrue(path.exists())
        self.assertTrue(any('preserved' in x['code'] for x in result['fixes']))

    def test_doctor_preserves_replacement_inode_with_same_identity(self):
        m = self.manifest
        m.start.pid_path = str(self.root / m.start.pid_path)
        path = Path(m.start.pid_path)
        self.active()
        original_guard = runner._guarded_remove
        def replace_then_remove(manifest, predicate):
            runner.write_pid_state(manifest, runner.read_pid_state(manifest))
            return original_guard(manifest, predicate)
        with patch.object(runner, 'capture_process_identity', return_value=None), patch.object(runner, 'live_process_group_members', return_value=[]), patch.object(runner, '_guarded_remove', side_effect=replace_then_remove), patch.object(ops, 'doctor', return_value={'ok': True}):
            result = ops.doctor_fix(m)
        self.assertTrue(path.exists(), result)
        self.assertTrue(any('preserved' in x['code'] for x in result['fixes']))

    def test_supervisors_skip_real_lock_contention(self):
        self.active()
        for caller in ('daemon', 'watchdog'):
            with self.subTest(caller=caller):
                sampled = threading.Event(); done = threading.Event(); result = []; errors = []
                def unhealthy(*args, **kwargs):
                    sampled.set(); return {'ok': False, 'status': 'critical'}
                def work():
                    try:
                        if caller == 'daemon':
                            result.append(ops.daemon(self.manifest, iterations=1, restart=True, health_mode=True))
                        else:
                            result.append(ops.watchdog(self.manifest, stop_on_breach=True))
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        done.set()
                with patch.object(runner, 'capture_process_identity', return_value=self.identity), patch.object(ops, 'health', side_effect=unhealthy), patch.object(ops, 'readiness_check', side_effect=lambda *a, **k: (sampled.set() or {'ready': False})), patch.object(runner, '_stop_locked', return_value={'ok': True, 'safe_to_start': True}) as stop, patch.object(ops, 'start_model', return_value={'ok': True, 'readiness': {'ready': True}}) as start:
                    with lifecycle.lifecycle_lock('rotate', self.manifest):
                        thread = threading.Thread(target=work); thread.start()
                        self.assertTrue(sampled.wait(3))
                        completed_while_locked = done.wait(1)
                    thread.join(5)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors, errors)
                self.assertTrue(completed_while_locked, 'supervisor waited for lifecycle lock')
                stop.assert_not_called(); start.assert_not_called()
                self.assertFalse(result[0]['ok'])

    def test_supervisors_preserve_new_generation_and_recheck_health(self):
        for caller in ('daemon', 'watchdog'):
            with self.subTest(caller=caller):
                self.active(); calls = []
                def change(*args, **kwargs):
                    calls.append(1)
                    if len(calls) == 1:
                        self.active(identity=ProcessIdentity(654322, 654322, 'successor'))
                    return {'ok': len(calls) > 1, 'status': 'ok' if len(calls) > 1 else 'critical', 'ready': len(calls) > 1}
                with patch.object(runner, 'capture_process_identity', side_effect=lambda pid: self.identity if pid == self.identity.leader_pid else ProcessIdentity(654322, 654322, 'successor')), patch.object(ops, 'health', side_effect=change), patch.object(ops, 'readiness_check', side_effect=change), patch.object(runner, '_stop_locked', return_value={'ok': True, 'safe_to_start': True}) as stop, patch.object(ops, 'start_model') as start:
                    result = ops.daemon(self.manifest, iterations=1, restart=True, health_mode=True) if caller == 'daemon' else ops.watchdog(self.manifest, stop_on_breach=True)
                stop.assert_not_called(); start.assert_not_called()
                self.assertFalse(result['ok'])

    def test_daemon_failed_stop_and_readiness_alone_never_recover(self):
        self.active()
        for stopped, started in (({'ok': False, 'safe_to_start': False}, {'ok': True, 'readiness': {'ready': True}}), ({'ok': True, 'safe_to_start': True}, {'ok': False, 'readiness': {'ready': True}})):
            with self.subTest(stopped=stopped):
                with patch.object(runner, 'capture_process_identity', return_value=self.identity), patch.object(ops, 'health', return_value={'ok': False, 'status': 'critical'}), patch.object(runner, '_stop_locked', return_value=stopped), patch.object(ops, 'start_model', return_value=started) as old_start, patch.object(runner, '_start_locked', return_value=started) as locked_start:
                    result = ops.daemon(self.manifest, iterations=1, restart=True, health_mode=True)
                self.assertFalse(result['ok'], result)
                if not stopped['safe_to_start']:
                    old_start.assert_not_called(); locked_start.assert_not_called()

    def test_daemon_does_not_count_unowned_ready_endpoint_as_recovery(self):
        self.active()
        with patch.object(runner, 'capture_process_identity', return_value=self.identity), patch.object(ops, 'health', return_value={'ok': False}), patch.object(runner, '_stop_locked', return_value={'ok': True, 'safe_to_start': True}), patch.object(runner, '_start_locked', return_value={'ok': True, 'started': True, 'pid': self.identity.leader_pid, 'readiness': {'ready': True}}), patch.object(runner, '_endpoint_owned', return_value={'owned': False}):
            result = ops.daemon(self.manifest, iterations=1, restart=True, health_mode=True)
        self.assertFalse(result['ok'], result)
        self.assertFalse(result['iterations'][0]['action']['endpoint_ownership']['owned'])

    def test_recheck_of_same_generation_prevents_unneeded_mutation(self):
        self.active()
        for caller in ('daemon', 'watchdog'):
            with self.subTest(caller=caller):
                responses = iter([False, True])
                def check(*args, **kwargs):
                    healthy = next(responses)
                    return {'ok': healthy, 'ready': healthy}
                with patch.object(runner, 'capture_process_identity', return_value=self.identity), patch.object(ops, 'health', side_effect=check), patch.object(ops, 'readiness_check', side_effect=check), patch.object(runner, '_stop_locked') as stop:
                    result = ops.daemon(self.manifest, iterations=1, restart=True, health_mode=True) if caller == 'daemon' else ops.watchdog(self.manifest, stop_on_breach=True)
                stop.assert_not_called()
                self.assertFalse(result['ok'])

    def test_supervisors_mutate_only_after_persistent_breach(self):
        for caller in ('daemon', 'watchdog'):
            with self.subTest(caller=caller):
                self.active()
                outcomes = iter([False, False, True])
                def check(*args, **kwargs):
                    healthy = next(outcomes)
                    return {'ok': healthy, 'ready': healthy}
                with patch.object(runner, 'capture_process_identity', return_value=self.identity), patch.object(ops, 'health', side_effect=check), patch.object(ops, 'readiness_check', side_effect=check), patch.object(runner, '_stop_locked', return_value={'ok': True, 'safe_to_start': True}) as stop, patch.object(runner, '_start_locked', return_value={'ok': True, 'started': True, 'pid': self.identity.leader_pid, 'readiness': {'ready': True}}) as start, patch.object(runner, '_endpoint_owned', return_value={'owned': True}):
                    result = ops.daemon(self.manifest, iterations=1, restart=True, health_mode=True) if caller == 'daemon' else ops.watchdog(self.manifest, stop_on_breach=True)
                stop.assert_called_once()
                self.assertEqual(start.call_count, int(caller == 'daemon'))
                self.assertEqual(result['ok'], caller == 'daemon')

    def test_rotation_real_target_interrupt_cleans_then_restores(self):
        target = self.make_manifest('target')
        spawned = []
        real_start, real_stop = runner._start_locked, runner._stop_locked
        restores = []
        def start(m, *args, **kwargs):
            if m is self.manifest:
                restores.append(True)
                return {'ok': True, 'started': True}
            return real_start(m, *args, **kwargs)
        def stop(m, timeout):
            return {'ok': True, 'safe_to_start': True} if m is self.manifest else real_stop(m, timeout)
        with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(spawned)), patch.object(runner, '_start_locked', side_effect=start), patch.object(runner, '_stop_locked', side_effect=stop), patch.object(runner, 'wait_ready', side_effect=KeyboardInterrupt('cancel')):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.rotate(self.manifest, target, readiness_timeout_sec=1, stop_timeout_sec=1)
        proc, _, _, _ = spawned[0]
        self.assertIsNone(capture_process_identity(proc.pid))
        self.assertFalse((self.root / target.start.pid_path).exists())
        self.assertTrue(restores)
        self.assertTrue(raised.exception.capstan_result['started'])
        self.assertTrue(raised.exception.capstan_result['cleanup']['group_death_certified'])
        self.assertTrue(raised.exception.capstan_recovery['cleanup_safe'])

    def test_rotation_real_target_interrupt_preserves_uncertain_blocker(self):
        target = self.make_manifest('target')
        spawned = []
        real_start, real_stop = runner._start_locked, runner._stop_locked
        restores = []
        def start(m, *args, **kwargs):
            if m is self.manifest:
                restores.append(True)
                return {'ok': True, 'started': True}
            return real_start(m, *args, **kwargs)
        def stop(m, timeout):
            return {'ok': True, 'safe_to_start': True} if m is self.manifest else real_stop(m, timeout)
        with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(spawned)), patch.object(runner, '_start_locked', side_effect=start), patch.object(runner, '_stop_locked', side_effect=stop), patch.object(runner, 'wait_ready', side_effect=KeyboardInterrupt('cancel')), patch.object(runner, 'terminate_process_identity', return_value=False):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.rotate(self.manifest, target, readiness_timeout_sec=1, stop_timeout_sec=1)
        proc, identity, _, _ = spawned[0]
        self.assertEqual(capture_process_identity(proc.pid), identity)
        self.assertTrue((self.root / target.start.pid_path).exists())
        self.assertFalse(restores)
        self.assertTrue(raised.exception.capstan_result['started'])
        self.assertFalse(raised.exception.capstan_result['cleanup']['group_death_certified'])
        self.assertFalse(raised.exception.capstan_recovery['attempted'])

    def test_rotation_cancel_during_stop_rechecks_before_restoring(self):
        target = self.make_manifest('target')
        count = []
        def stop(m, timeout):
            count.append(m)
            if len(count) == 1:
                raise KeyboardInterrupt('stop')
            return {'ok': False, 'safe_to_start': False}
        with patch.object(runner, '_inspect', return_value={'status': 'absent'}), patch.object(runner, '_stop_locked', side_effect=stop), patch.object(runner, '_start_locked') as start:
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.rotate(self.manifest, target, readiness_timeout_sec=1, stop_timeout_sec=1)
        start.assert_not_called()
        self.assertFalse(raised.exception.capstan_recovery['cleanup_safe'])

    def test_receipt_matches_actual_launch_across_callers_and_env_drift(self):
        m = self.manifest
        interpreter = self.root / 'virtualenv-python'
        interpreter.symlink_to(sys.executable)
        m.start.command[0] = str(interpreter)
        a = self.root / 'a'; b = self.root / 'b'; a.mkdir(); b.mkdir()
        old = Path.cwd(); captured = []
        try:
            os.chdir(a); first = receipt.candidate_binding(m)
            os.chdir(b); second = receipt.candidate_binding(m)
            with patch.dict(os.environ, {'CONSISTENCY_DRIFT': 'changed'}):
                drift = receipt.candidate_binding(m)
            with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(captured)):
                result = runner.start(m)
        finally:
            os.chdir(old)
        self.assertEqual(first['candidate_fingerprint'], second['candidate_fingerprint'])
        self.assertNotEqual(first['candidate_fingerprint'], drift['candidate_fingerprint'])
        self.assertTrue(result['started'], result)
        proc, identity, args, kwargs = captured[0]
        self.assertEqual(args[0], m.start.command)
        self.assertEqual(args[0][0], str(interpreter))
        self.assertEqual(kwargs['cwd'], str(self.root))
        self.assertEqual(first['launch_fingerprint'], command_cwd_fingerprint(args[0], kwargs['cwd']))
        encoded = json.dumps(kwargs['env'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        self.assertEqual(first['environment_fingerprint'], hashlib.sha256(encoded).hexdigest())
        m.start.cwd = str(self.root)
        self.assertEqual(first['candidate_fingerprint'], receipt.candidate_binding(m)['candidate_fingerprint'])

    def test_interrupt_readiness_cleans_real_spawn_and_records_started(self):
        captured = []
        with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(captured)), patch.object(runner, 'wait_ready', side_effect=KeyboardInterrupt('cancel')):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.start(self.manifest, wait=True)
        proc, identity, _, _ = captured[0]
        self.assertIsNone(capture_process_identity(proc.pid))
        self.assertFalse((self.root / self.manifest.start.pid_path).exists())
        self.assertTrue(getattr(raised.exception, 'capstan_result', {}).get('started'))

    def test_interrupt_before_identity_capture_retains_pending_fence(self):
        spawned = []
        with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(spawned)), patch.object(runner, '_capture_new_identity', side_effect=KeyboardInterrupt('capture')):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.start(self.manifest)
        proc, _, _, _ = spawned[0]
        self.assertIsNone(capture_process_identity(proc.pid))
        self.assertTrue(raised.exception.capstan_result['started'])
        self.assertTrue(raised.exception.capstan_result['durable_pending'])
        self.assertTrue((self.root / self.manifest.start.pid_path).exists())

    def test_interrupt_in_popen_constructor_keeps_pending_fence(self):
        with patch.object(runner.subprocess, 'Popen', side_effect=KeyboardInterrupt('constructor')):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.start(self.manifest)
        self.assertTrue(raised.exception.capstan_result['spawn_uncertain'])
        self.assertTrue(raised.exception.capstan_result['durable_pending'])
        self.assertTrue((self.root / self.manifest.start.pid_path).exists())

    def test_rotation_interrupt_recovers_only_after_certified_target_cleanup(self):
        target = self.make_manifest('target')
        for safe in (True, False):
            with self.subTest(safe=safe):
                def stop(m, timeout):
                    return {'ok': True, 'safe_to_start': True} if m is self.manifest else {'ok': safe, 'safe_to_start': safe}
                def start(m, *args, **kwargs):
                    if m is target:
                        raise KeyboardInterrupt('cancel')
                    return {'ok': True, 'started': True}
                with patch.object(runner, '_inspect', return_value={'status': 'absent'}), patch.object(runner, '_stop_locked', side_effect=stop), patch.object(runner, '_start_locked', side_effect=start) as starts:
                    with self.assertRaises(KeyboardInterrupt):
                        runner.rotate(self.manifest, target, readiness_timeout_sec=1, stop_timeout_sec=1)
                self.assertEqual(sum(call.args[0] is self.manifest for call in starts.call_args_list), int(safe))

    def test_rotation_interrupt_during_handoff_recovers(self):
        target = self.make_manifest('target')
        with patch.object(runner, '_inspect', return_value={'status': 'absent'}), patch.object(runner, '_stop_locked', return_value={'ok': True, 'safe_to_start': True}), patch.object(runner, '_start_locked', return_value={'ok': True, 'started': True}) as start, patch.object(runner, '_rotation_handoff_locked', side_effect=KeyboardInterrupt('handoff')):
            with self.assertRaises(KeyboardInterrupt):
                runner.rotate(self.manifest, target, readiness_timeout_sec=1, stop_timeout_sec=1)
        self.assertEqual([call.args[0] for call in start.call_args_list], [target, self.manifest])

    def test_rotation_cancellation_between_phases_recovers_real_target(self):
        target = self.make_manifest('target')
        for boundary in ('if stop_current.get(', 'if target_start.get('):
            with self.subTest(boundary=boundary):
                captured, restores, reached = [], [], []
                real_start, real_stop = runner._start_locked, runner._stop_locked

                def start(manifest, *args, **kwargs):
                    if manifest is self.manifest:
                        for proc, identity, _, _ in captured:
                            self.assertNotEqual(capture_process_identity(proc.pid), identity)
                        self.assertFalse((self.root / target.start.pid_path).exists())
                        restores.append(True)
                        return {'ok': True, 'started': True}
                    return real_start(manifest, *args, **kwargs)

                def stop(manifest, timeout):
                    if manifest is self.manifest:
                        return {'ok': True, 'safe_to_start': True}
                    return real_stop(manifest, timeout)

                source, first_line = inspect.getsourcelines(runner._rotate_locked)
                line = first_line + next(i for i, text in enumerate(source) if text.strip().startswith(boundary))

                def interrupt(frame, event, arg):
                    if frame.f_code is runner._rotate_locked.__code__ and event == 'line' and frame.f_lineno == line:
                        reached.append(boundary)
                        sys.settrace(None)
                        raise KeyboardInterrupt('between rotation phases')
                    return interrupt

                previous_trace = sys.gettrace()
                try:
                    with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(captured)), patch.object(runner, '_start_locked', side_effect=start), patch.object(runner, '_stop_locked', side_effect=stop), patch.object(runner, 'wait_ready', return_value={'ready': True}), patch.object(runner, '_endpoint_owned', return_value={'owned': True}):
                        sys.settrace(interrupt)
                        with self.assertRaises(KeyboardInterrupt) as raised:
                            runner.rotate(self.manifest, target, readiness_timeout_sec=1, stop_timeout_sec=1)
                finally:
                    sys.settrace(previous_trace)
                self.assertEqual(reached, [boundary])
                self.assertEqual(len(captured), int(boundary.startswith('if target_start')))
                self.assertEqual(restores, [True])
                self.assertTrue(raised.exception.capstan_recovery['cleanup_safe'])

    def test_readiness_error_cleanup_cancellation_propagates_with_blocker(self):
        captured = []
        cancellation = KeyboardInterrupt('cancel during cleanup')
        with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(captured)), patch.object(runner, 'wait_ready', side_effect=RuntimeError('readiness failed')), patch.object(runner, 'terminate_process_identity', side_effect=cancellation):
            with self.assertRaises(KeyboardInterrupt) as raised:
                runner.start(self.manifest, wait=True)
        self.assertIs(raised.exception, cancellation)
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        result = raised.exception.capstan_result
        self.assertTrue(result['started'])
        self.assertFalse(result['cleanup']['group_death_certified'])
        self.assertTrue(result['durable_blocker'])
        self.assertTrue((self.root / self.manifest.start.pid_path).exists())
        proc, identity, _, _ = captured[0]
        self.assertEqual(capture_process_identity(proc.pid), identity)

    def test_readiness_phase_cancellation_between_calls_cleans_real_spawn(self):
        captured, reached = [], []
        source, first_line = inspect.getsourcelines(runner._start_locked)
        line = first_line + next(i for i, text in enumerate(source) if text.strip().startswith('result.update({"readiness": readiness'))

        def interrupt(frame, event, arg):
            if frame.f_code is runner._start_locked.__code__ and event == 'line' and frame.f_lineno == line:
                reached.append(True)
                sys.settrace(None)
                raise KeyboardInterrupt('after readiness call')
            return interrupt

        previous_trace = sys.gettrace()
        try:
            with patch.object(runner.subprocess, 'Popen', new=self.capture_spawn(captured)), patch.object(runner, 'wait_ready', return_value={'ready': False}):
                sys.settrace(interrupt)
                with self.assertRaises(KeyboardInterrupt) as raised:
                    runner.start(self.manifest, wait=True)
        finally:
            sys.settrace(previous_trace)
        self.assertEqual(reached, [True])
        proc, identity, _, _ = captured[0]
        self.assertNotEqual(capture_process_identity(proc.pid), identity)
        self.assertFalse((self.root / self.manifest.start.pid_path).exists())
        self.assertTrue(raised.exception.capstan_result['started'])
