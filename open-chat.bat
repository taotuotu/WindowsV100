@echo off
setlocal
set "chat_url=%~1"
if not defined chat_url set "chat_url=http://127.0.0.1:8110/"
start "" "%chat_url%"
endlocal
