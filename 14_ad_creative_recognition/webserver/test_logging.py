import colorlog
import logging

# logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter("%(asctime)s %(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt='%d-%H:%M:%S'))
logger = colorlog.getLogger('root')
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.error("hahahahha")
logger.info("hahahahha")
