#!/bin/bash
# Simple one-liner test for LLM loading
pipenv run python -c "from src.llm import OpenSourceLLM; m=OpenSourceLLM(); print('OK', m.generate_reply('Hello'))"
