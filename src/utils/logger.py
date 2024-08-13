import logging
import sys
import os
  
def setup_logging(args, include_host=False):
    args.log_path = os.path.join(
        args.log_path, 
        f'{args.current_time}_{args.feature_extractor}_{args.classifier}' + (f'_{args.extra_log}' if args.extra_log else '')
    )
    os.makedirs(args.log_path, exist_ok = True)
    log_filename = 'out.log'
    log_path = os.path.join(args.log_path, log_filename)
    if os.path.exists(log_path):
        print("Error. Experiment already exists.")
        return -1
    args.log_level = logging.DEBUG if args.debug else logging.INFO

    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(args.log_level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(args.log_level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_path :
        file_handler = logging.FileHandler(filename=log_path )
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
