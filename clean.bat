@echo off
set dirs2del="./,bin,Windows"
:DEL_LOOP
for /f "Tokens=1,* Delims=," %%a in (%dirs2del%) do (
echo clearing %%a
if exist "%%a\x64" rd "%%a\x64" /s /q
if exist "%%a\Release" rd "%%a\Release" /s /q
if exist "%%a\Debug" rd "%%a\Debug" /s /q
if exist "%%a\*.exp" del "%%a\*.exp" /f /s /q
if exist "%%a\*.pdb" del "%%a\*.pdb" /f /s /q
if exist "%%a\*.VC.db" del "%%a\*.VC.db" /f /s /q
set dirs2del="%%b"
goto DEL_LOOP
)
pause