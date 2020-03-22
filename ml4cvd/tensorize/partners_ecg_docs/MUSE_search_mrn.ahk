!r::

; Start this script with sublime text on top of MUSE Editor
; MUSE Editor must be configured for Ctrl + A to select test list
; MUSE Editor must be configured for Ctrl + P to print selected tests
; MUSE Editor should be on print retrieve screen
; MUSE Editor export folder should be first in list
; In retrieve vie of MUSE Editor, select ECG type for search
; Place the cursor at the first line of the list of MRNs in Sublime Text

SetKeyDelay, 250

Loop {

; Cut the MRN from the list
Send, ^x

; Check if anything aside from newline was cut
If InStr(Clipboard, "`r`n") == 1
	Break

; Switch to MUSE Editor (Alt Tab)
Send, !`t

; Select retrieve search
Send, ^r

; Delete the previous search
SetKeyDelay, 10
Send, {Right 10}
Send, {BackSpace 10}
SetKeyDelay, 250

; Paste in new search
Send, ^v

; Search for MRN (Enter)
Send, `n

; Wait for search to return
; TODO make this detect change in pixel
Sleep, 1000

; Select all search results
Send, ^a

; Print search results
Send, ^p

; Wait for screen to popup
Sleep, 500

; Print using default options (Enter)
Send, `n

; Wait for MUSE Editor to unfreeze
Loop {
	; Search for a black pixel in the space where window goes blank/freezes
	PixelSearch, Px, Py, 503, 265, 1351, 741, 0x000000, Fast
	If Not ErrorLevel {
		Break
	}

	Sleep, 250
}

; Go back to Sublime to repeat (Alt Tab)
Send, !`t

}

return