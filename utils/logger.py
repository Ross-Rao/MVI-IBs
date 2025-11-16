# python import
import traceback
import functools
# package import
# local import


__all__ = ["log_exception"]


def log_exception(logger):
    """
    装饰器：捕获函数执行中的异常并记录堆栈信息。
    可以传入自定义的 logger，如果没有传入，则使用默认的 logging 模块。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Exception occurred in {func.__name__}: {e}")
                logger.error("Traceback:\n" + traceback.format_exc())
                # 可以选择在这里重新抛出异常，或者返回一个默认值
                raise  # 重新抛出异常
        return wrapper
    return decorator
