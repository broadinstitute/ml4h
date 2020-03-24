#NoEnv  ; Recommended for performance and compatibility with future AutoHotkey releases.
; #Warn  ; Enable warnings to assist with detecting common errors.
SendMode Input  ; Recommended for new scripts due to its superior speed and reliability.
SetWorkingDir %A_ScriptDir%  ; Ensures a consistent starting directory.


!j::


; Script setup
; Set coordinates relative to screen
; This may allow the script to continue if small windows pop up
CoordMode, Mouse, Screen
CoordMode, Pixel, Screen
WinMaximize, ahk_exe MUSEEditor.exe
count := 0
extracted := 0


; Open file to read input from, line by line
Loop, Read, C:\Users\MuseAdmin\Desktop\blake08_mrns_to_extract.csv
{
    ; Switch to MUSE Editor
    WinActivate, ahk_exe MUSEEditor.exe

    ; Remove previous input from search box
    Loop, 20 {
       Click, 135, 830
       Send, {Delete}
       Send, {Backspace}
    }

    ; Enter new input to search box
    Click, 135, 830
    Send, %A_LoopReadLine%


    ; SCRIPT OTHER SEARCH CRITERIA HERE


    ; Click search button
    Click, 70, 1010

    ; Select all items in search result list
    Click, 86, 26
    Click, 86, 96

    ; Print list
    Click, 415, 55
    Sleep, 750
    Click, 1185, 800
    extracted++

    ; Wait for MUSE Editor to unfreeze
    Loop {
        ; Search for a black pixel in the space where window goes blank/freezes
        PixelSearch, Px, Py, 515, 265, 1290, 660, 0x000000, Fast
        If Not ErrorLevel {
            Break
        }
        Sleep, 100
    }

    ; Reset MUSE Editor after 1000 searches
    count++
    If (count > 1000)
    {
        Sleep, 20000
        WinClose, ahk_exe MUSEEditor.exe
        Run "C:\Program Files (x86)\MUSE\MUSEEditor.exe", C:\Program Files (x86)\MUSE
        Sleep, 1000
        count := 0
        Sleep, 20000
    }
}

MsgBox, Extracted %extracted% MRNs
Return


!r::
MsgBox, Script Reloaded
Reload
Return


!Esc::ExitApp
