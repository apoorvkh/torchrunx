import pytest
from unittest.mock import patch, MagicMock
from src.torchrunx.__main__ import main

#@patch('src.tyro.cli')
@patch('src.__main__.dill.loads')
@patch('src.__main__.dist.init_process_group')
@patch('src.__main__.start_processes')
def test_main(mock_start_processes, mock_init_process_group, mock_dill_loads):
    mock_dill_loads.return_value = lambda x: x * 2

    mock_ctx = MagicMock()
    mock_ctx.wait.return_value.return_values = [4, 4] 
    mock_start_processes.return_value = mock_ctx

    result = main('127.0.0.1', 12345)

    #mock_init_process_group.assert_called_with(backend="nccl", world_size=1, rank=0) # this won't occur since src.__main__.start_processes is patched
    mock_start_processes.assert_called()
    mock_ctx.wait.assert_called()
    
    assert result == [4, 4]
