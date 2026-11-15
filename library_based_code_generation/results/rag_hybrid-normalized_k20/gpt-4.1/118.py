```python
from psyclone.transformations import (
    RedundantComputationTrans,
    AsyncHaloExchangeTrans,
    OMPColourTrans,
    InlineTrans,
    TransformationError
)
from psyclone.constants import VALID_LOOP_TYPES

# Boolean variables to control transformations
APPLY_REDUNDANT_COMP = True
APPLY_ASYNC_HALO_EXCHANGE = True
APPLY_OMP_COLOUR = True
APPLY_INTRINSIC_INLINE = True

def apply_transformations(psy):
    redundant_comp_trans = RedundantComputationTrans()
    async_halo_exchange_trans = AsyncHaloExchangeTrans()
    omp_colour_trans = OMPColourTrans()
    inline_trans = InlineTrans()

    for invoke in psy.invokes.invoke_list:
        schedule = invoke.schedule
        if not schedule:
            continue
        try:
            if APPLY_REDUNDANT_COMP:
                for loop in schedule.loops():
                    if loop.loop_type in VALID_LOOP_TYPES:
                        redundant_comp_trans.apply(loop)
            if APPLY_ASYNC_HALO_EXCHANGE:
                for node in schedule.walk(schedule.__class__.HaloExchange):
                    async_halo_exchange_trans.apply(node)
            if APPLY_OMP_COLOUR:
                for loop in schedule.loops():
                    if loop.loop_type in VALID_LOOP_TYPES:
                        omp_colour_trans.apply(loop)
            if APPLY_INTRINSIC_INLINE:
                for call in schedule.walk(schedule.__class__.IntrinsicCall):
                    inline_trans.apply(call)
        except TransformationError as err:
            print(f"TransformationError in invoke '{invoke.name}': {err}")
    return psy
```