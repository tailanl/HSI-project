"""Versioned store authority wiring; original v1 is untouched.

Only the two admission imports change. All original transaction, CAS, replay,
promotion and quarantine method bytecode remains unchanged. A private globals
dictionary gives this subclass a distinct store/schema/authority policy, so v1
cannot open its files. No process-global module registry or callback is used.
"""
from pathlib import Path
import inspect
import textwrap
import types

import memory_store as original
from memory_common import artifact, require

HERE = Path(__file__).resolve().parent
ORIGINAL_SHA256 = '2a5c78e13593950201da951b23b55e91ea65d9821c66ac2c443ae59ea6d69974'
require(artifact(original.__file__)['sha256'] == ORIGINAL_SHA256, 'Frozen v1 store changed')
AUTHORITY_SOURCES = tuple(artifact(HERE/name) for name in (
    'memory_store_v2.py', 'episode_evidence_v2.py', 'extract_local_experience_v2.py',
    'motion_semantic_contract_v1.py', 'memory_common.py', 'memory_schema.py'))
_scope = dict(vars(original))
_scope.update(__file__=__file__, STORE_SCHEMA='p555.memory_store.v2',
    SNAPSHOT_SCHEMA='p555.positive_memory_snapshot.v2', CYCLE_SCHEMA='p555.frozen_memory_cycle.v2',
    EVENT_SCHEMA='p555.memory_journal_event.v2', POLICY={**original.POLICY,
        'fixed_admission_authority':'episode_evidence_v2.admit_episode',
        'authority_source_bindings':list(AUTHORITY_SOURCES)})


def _clone(function):
    result=types.FunctionType(function.__code__,_scope,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    result.__doc__=function.__doc__
    return result


for _name,_value in vars(original).items():
    if isinstance(_value,types.FunctionType) and _value.__globals__ is vars(original):
        _scope[_name]=_clone(_value)

_methods={}
for _name,_value in vars(original.MemoryStore).items():
    if isinstance(_value,types.FunctionType):
        _methods[_name]=_clone(_value) if _value.__globals__ is vars(original) else _value

for _name in ('record_attempt','commit_cycle'):
    _source=textwrap.dedent(inspect.getsource(getattr(original.MemoryStore,_name)))
    _old='from episode_evidence import admit_episode'
    require(_source.count(_old)==1,'Expected exactly one fixed authority import')
    _source=_source.replace(_old,'from episode_evidence_v2 import admit_episode')
    exec(compile(_source,__file__,'exec'),_scope)
    _methods[_name]=_scope[_name]

MemoryStore=type('MemoryStore',(original.MemoryStore,),{**_methods,'__module__':__name__,
    '__doc__':__doc__})
group_key=_scope['group_key']


def require_authority_sources():
    """Each public producer entry also calls this before and after a read."""
    require(tuple(artifact(row['path']) for row in AUTHORITY_SOURCES)==AUTHORITY_SOURCES,
        'Fixed v2 admission/store/extraction source changed after import')
