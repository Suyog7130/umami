import logging
import datetime
import argparse
import sys
import os
import inspect
import types

def init_logging(args, log_dir='logs', write_to_file=True):
    """
    Smart logging initializer that auto-discovers imported project modules 
    and applies hierarchical verbosity scaling.

    This sets-up the `logging` module, so no return `logger` object is necessary!

    NOTE: Do not use the returned logger, instead use `logging.getLogger(__name__)` 
    in your modules to get the correct logger for that module, and then simply call
    this function from `__main__` to initialize the logging system.
    
    Levels:
    --trace   : DEBUG everywhere (Current file + Imports + Transitive deps)
    --debug   : DEBUG current file + Direct Imports. INFO for Transitive.
    --describe: DEBUG current file + INFO on imports. WARNING for others.
    --verbose : INFO current file + Direct Imports. WARNING for others.
    Default   : INFO current file. WARNING for imports.
    --quiet   : WARNING/ERROR only everywhere.
    """
    if type(args) is argparse.ArgumentParser:
        args = args.parse_args()
    
    # 1. Define Level Hierarchy
    # format: (Root/Current File, Direct Imports, Transitive/Deep Libs)
    if args.trace:
        levels = (logging.DEBUG, logging.DEBUG, logging.DEBUG)
    elif args.debug:
        levels = (logging.DEBUG, logging.DEBUG, logging.INFO)
    elif args.describe:
        levels = (logging.DEBUG, logging.INFO, logging.WARNING)
    elif args.verbose:
        levels = (logging.INFO, logging.INFO, logging.WARNING)
    elif args.quiet:
        levels = (logging.INFO, logging.WARNING, logging.WARNING)
    else:
        # Default: Standard Production
        levels = (logging.WARNING, logging.WARNING, logging.ERROR)
        
    root_level, direct_import_level, deep_import_level = levels

    # 2. Configure Handlers
    # log_format = '%(asctime)s~%(name)-15s~%(levelname)-8s: %(message)s'
    log_format = '%(asctime)s ~ %(name)s ~ %(levelname)s : %(message)s'
    date_format = '%H:%M:%S'
    
    # Console: Dynamic verbosity based on flags
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(root_level)
    
    if write_to_file:
        os.makedirs(log_dir, exist_ok=True)
        fname = os.path.join(log_dir, f'session_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
        file_handler = logging.FileHandler(fname, mode='w')
        file_handler.setLevel(root_level)   # -- saved logs should be as detailed as possible!

    # 3. Initialize Root Logger
    # We set basicConfig to the lowest logical level so handlers can filter up
    logging.basicConfig(
        level=logging.NOTSET, 
        format=log_format,
        datefmt=date_format,
        handlers=[stream_handler, file_handler] if write_to_file else [stream_handler],
        force=True  # Force reconfiguration in case logging was already set up
    )
    
    # 4. Auto-Discovery Engine: Find caller's imports
    # Get the frame of the caller (the script calling init_logging)
    caller_frame = inspect.stack()[1]
    caller_globals = caller_frame[0].f_globals
    
    # Identify "Internal" modules (exclude venv, stdlib, site-packages)
    base_prefix = sys.base_prefix
    project_root = os.getcwd() # Assuming script is run from project root

    # NOTE: `caller_globals` only contains the caller's specific imported class or function names,
    # and the actual file name of the modules. Thus, we get the issue of `umamipe.py` being set to
    # INFO for both `--verbose` and `--quiet` mode, because only the specific imported `MLWaveformGenerator`
    # is detected as an import, and not the entire `umamipe` module. So, we switch to `sys.modules`!
    
    for name, module in sys.modules.items():
        # We only care about Modules
        if isinstance(module, types.ModuleType) and hasattr(module, '__file__'):
            # Skip modules without a file (built-ins)
            if not module.__file__: continue
            
            # Normalize path to handle cross-platform slashes
            mod_path = os.path.normpath(module.__file__)
            
            # LOGIC: If it lives in project_root and NOT in site-packages -> It's Ours
            is_internal = (
                mod_path.startswith(project_root) and 
                'site-packages' not in mod_path and
                'dist-packages' not in mod_path
            )
            
            if is_internal:
                # Apply "Direct Import" Level
                logging.getLogger(module.__name__).setLevel(direct_import_level)
                
                # Apply "Transitive/Deep" Level (Silence submodules)
                # This prevents 'my_module.heavy_logic' from spamming if only 'my_module' is imported
                # We iterate children or just set the base to propagate correctly
                pass 
            else:
                # Silence Third-Party Libs (matplotlib, boto3, etc) unless Tracing
                logging.getLogger(module.__name__).setLevel(deep_import_level)

    # Remove Bilby's native logging handlers and set it to propagate to our root logger
    bilby_logger = logging.getLogger('bilby')
    bilby_logger.handlers = []   # STRIP its private handlers
    bilby_logger.propagate = True   # ENABLE propagation to our root logger, so it follows our configured levels
    bilby_logger.setLevel(root_level)  # Set to root level to ensure it follows our hierarchy
    
    # NOTE: Some issue with `nessai` and `nessai.sampler.xxx` loggers, so better to not propagate them!
    logging.getLogger("nessai").propagate = True

    # 5. Manual Overrides for Noisy Libraries
    # Even in verbose modes, these are often too noisy
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger('dynesty').setLevel(logging.WARNING)

    # 6. Return a logger for the current file
    # If called from main.py, __name__ here is the utility module, so we grab the caller's name
    caller_logger_name = caller_globals.get('__name__', '__main__')
    logger = logging.getLogger(caller_logger_name)
    logger.setLevel(root_level)
    return logger


def init_verbosity_args(parser: argparse.ArgumentParser = None) -> argparse.ArgumentParser:
    """
    Attaches a mutually exclusive logging verbosity group to an existing parser instance.
    Import this from your utils module across projects.

    Allowed verbosity levels:
        ### trace
            Ultra-deep debugging (Current file + Imports + Transitive deps)
        ### debug
            Standard debugging (Current file + Direct internal imports)
        ### describe
            Standard debugging (Current file + INFO on imports)
        ### verbose
            Surface level runtime details (Root script & 1st tier imports)
        ### quiet
            Suppress normal outputs; display warnings and errors only
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Parser with mutually exclusive verbosity arguments.")

    # Create the mutually exclusive group directly on the provided parser
    verbosity_group = parser.add_mutually_exclusive_group()

    verbosity_group.add_argument(
        "--trace", 
        action="store_true", 
        help="Ultra-deep debugging (Current file + Imports + Transitive deps)"
    )
    verbosity_group.add_argument(
        "--debug", 
        action="store_true", 
        help="Standard debugging (Current file + Direct internal imports)"
    )
    verbosity_group.add_argument(
        "--describe", 
        action="store_true", 
        help="Standard debugging (Current file + INFO on imports)"
    )
    verbosity_group.add_argument(
        "-v",
        "--verbose", 
        action="store_true", 
        help="Surface level runtime details (Root script & 1st tier imports)"
    )
    verbosity_group.add_argument(
        "--quiet", 
        action="store_true", 
        help="Suppress normal outputs; display warnings and errors only"
    )
    return parser
