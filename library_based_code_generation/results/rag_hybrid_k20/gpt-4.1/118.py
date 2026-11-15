```python
from psyclone.transformations import (
    RedundantComputationTrans,
    AsyncHaloExchangeTrans,
    OMPColourTrans,
    InlineTrans,
    TransformationError
)
from psyclone.psyGen import PSyFactory

# Boolean variables to control transformations
APPLY_REDUNDANT_COMPUTATION = True
APPLY_ASYNC_HALO_EXCHANGE = True
APPLY_OMP_COLOURING = True
APPLY_INTRINSIC_INLINING = True

def apply_transformations(psy):
    redundant_comp_trans = RedundantComputationTrans()
    async_halo_exchange_trans = AsyncHaloExchangeTrans()
    omp_colour_trans = OMPColourTrans()
    inline_trans = InlineTrans()

    for invoke in psy.invokes.invoke_list:
        schedule = invoke.schedule
        try:
            if APPLY_REDUNDANT_COMPUTATION:
                for loop in schedule.loops():
                    redundant_comp_trans.apply(loop)
            if APPLY_ASYNC_HALO_EXCHANGE:
                for node in schedule.walk(schedule.__class__.HaloExchange):
                    async_halo_exchange_trans.apply(node)
            if APPLY_OMP_COLOURING:
                for loop in schedule.loops():
                    omp_colour_trans.apply(loop)
            if APPLY_INTRINSIC_INLINING:
                for call in schedule.calls():
                    inline_trans.apply(call)
        except TransformationError as err:
            print(f"TransformationError in invoke '{invoke.name}': {err}")
    return psy
```