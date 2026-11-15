```python
from psyclone.transformations import (
    RedundantComputationTrans,
    AsyncHaloExchangeTrans,
    OMPColourTrans,
    IntrinsicInlineTrans,
    TransformationError
)
from psyclone.core import AccessType

# Boolean variables to control transformations
APPLY_REDUNDANT_COMP = True
APPLY_ASYNC_HALO_EXCHANGE = True
APPLY_OMP_COLOUR = True
APPLY_INTRINSIC_INLINE = True

def apply_transformations(psy):
    redundant_comp_trans = RedundantComputationTrans()
    async_halo_exchange_trans = AsyncHaloExchangeTrans()
    omp_colour_trans = OMPColourTrans()
    intrinsic_inline_trans = IntrinsicInlineTrans()

    for invoke in psy.invokes.invoke_list:
        schedule = invoke.schedule
        try:
            if APPLY_REDUNDANT_COMP:
                for loop in schedule.loops():
                    if redundant_comp_trans.validate(loop):
                        schedule, _ = redundant_comp_trans.apply(loop)
            if APPLY_ASYNC_HALO_EXCHANGE:
                for node in schedule.walk(schedule.halo_exchange_class):
                    if async_halo_exchange_trans.validate(node):
                        schedule, _ = async_halo_exchange_trans.apply(node)
            if APPLY_OMP_COLOUR:
                for loop in schedule.loops():
                    if omp_colour_trans.validate(loop):
                        schedule, _ = omp_colour_trans.apply(loop)
            if APPLY_INTRINSIC_INLINE:
                for call in schedule.walk(schedule.intrinsic_call_class):
                    if intrinsic_inline_trans.validate(call):
                        schedule, _ = intrinsic_inline_trans.apply(call)
            invoke.schedule = schedule
        except TransformationError as err:
            print(f"TransformationError in invoke '{invoke.name}': {err}")
    return psy
```