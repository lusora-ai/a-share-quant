import logging
import tempfile
import os
from pathlib import Path
from ashare_quant.utils.logging import setup_logger

def test_setup_logger():
    logger = setup_logger("test_logger", level="DEBUG", console=False)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG

def test_logger_file_output():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = os.path.join(tmpdir, "test.log")
        logger = setup_logger("file_logger", level="INFO", log_file=log_file, console=False)
        logger.info("Hello Quant World!")
        
        assert os.path.exists(log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Hello Quant World!" in content
            assert "[INFO]" in content
            
        # Close handlers so Windows can delete temp file
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

