```python
from psyclone.transformations import (
    RedundantComputationTrans,
    AsyncHaloExchangeTrans,
    OMPColourTrans,
    IntrinsicInlineTrans,
    TransformationError
)
from psyclone.psyGen import PSyFactory

# Boolean variables to control transformations
apply_redundant_computation = True
apply_async_halo_exchange = True
apply_omp_colouring = True
apply_intrinsic_inlining = True

def apply_transformations(psy):
    redundant_comp_trans = RedundantComputationTrans()
    async_halo_trans = AsyncHaloExchangeTrans()
    omp_colour_trans = OMPColourTrans()
    intrinsic_inline_trans = IntrinsicInlineTrans()

    for invoke in psy.invokes.invoke_list:
        schedule = invoke.schedule
        try:
            if apply_redundant_computation:
                for loop in schedule.loops():
                    redundant_comp_trans.apply(loop)
            if apply_async_halo_exchange:
                for node in schedule.walk(schedule.halo_exchange_class):
                    async_halo_trans.apply(node)
            if apply_omp_colouring:
                for loop in schedule.loops():
                    omp_colour_trans.apply(loop)
            if apply_intrinsic_inlining:
                for call in schedule.walk(schedule.intrinsic_call_class):
                    intrinsic_inline_trans.apply(call)
        except TransformationError as err:
            print(f"TransformationError in invoke '{invoke.name}': {err}")
    return psy
```