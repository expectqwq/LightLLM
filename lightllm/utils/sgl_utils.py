from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
try:
    import sgl_kernel

    sgl_ops = sgl_kernel
    sgl_allreduce_ops = sgl_ops.allreduce
    HAS_SGL_KERNEL = True
except:
    sgl_ops = None
    sgl_allreduce_ops = None
    HAS_SGL_KERNEL = False
    logger.warning(
        "sgl_kernel is not installed, you can't use the api of it. \
                   You can solve it by running `pip install sgl_kernel`."
    )

try:
    from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

    merge_state_v2 = sgl_ops.merge_state_v2
except ImportError as sgl_fa3_error:
    # SenseNova ships a patched FA3 extension with Neo image-token masking,
    # but it does not require the optional sgl_kernel Python package.  Using
    # only sgl_kernel as the availability probe made LightLLM's auto selector
    # reject this working extension and silently fall back to Triton.
    try:
        from flash_attn_interface import flash_attn_varlen_func
        from flash_attn_interface import flash_attn_with_kvcache as _flash_attn_with_kvcache

        def flash_attn_with_kvcache(*args, **kwargs):
            # The sgl_kernel wrapper accepts attention sinks.  The standalone
            # FA3 interface does not; dropping None preserves the ordinary
            # Neo path while refusing to silently change models that use
            # non-empty sinks.
            sinks = kwargs.pop("sinks", None)
            if sinks is not None:
                raise NotImplementedError("standalone FA3 does not support attention sinks")
            return _flash_attn_with_kvcache(*args, **kwargs)

        merge_state_v2 = None
        logger.info("using standalone flash_attn_interface as the FA3 backend")
    except ImportError:
        flash_attn_varlen_func = None
        flash_attn_with_kvcache = None
        merge_state_v2 = None
        logger.warning(
            "Neither sgl_kernel FA3 nor standalone flash_attn_interface is available. "
            f"sgl_kernel error: {sgl_fa3_error}"
        )
