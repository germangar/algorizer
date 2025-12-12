
# --- STATUS LINE MANAGEMENT ---
# import aioconsole
import sys
from colorama import init as colorama_init

def init_colorama():
    colorama_init() 

status_line_text = ''

'''
status_line_active = False

# without using a task. Just delete before running parseCandleUpdateMulti and write it back after
def delete_last_line( count=1):
    """
    Deletes the last line of text printed to the console.
    Requires colorama to be initialized for Windows compatibility.
    """
    global status_line_active
    

    # Code Sequence	Name	Action
    # \033[ or \x1B[	CSI (Control Sequence Introducer)	Signals the start of an ANSI command.
    # \033[1A	CUU (Cursor Up)	Moves the cursor up one line.
    # \r	CR (Carriage Return)	Moves the cursor to the beginning of the current line.
    # \033[K	EL (Erase in Line)	Clears all characters from the current cursor position to the end of the line.
    # "\033[1B" move down by 1 lines
    
    # 1. Move cursor UP one line: '\033[1A'
    # 2. Move cursor to start of line: '\r'
    # 3. Clear line from cursor to end: '\033[K'
    
    # Combined sequence:
    if status_line_active:
        assert( count > 0 )
        
        for i in range(count):
            sys.stdout.write(f"\033[1A\r\033[K")
        sys.stdout.flush()
    status_line_active = False

def print_status_line():
    global status_line_active
    if not status_line_active : 
        print( f"{status_line_text}")
    status_line_active = True
'''


# --- NEW ABSOLUTE POSITIONING CONSTANTS ---
# \033[s: Save Cursor Position (SCP)
# \033[u: Restore Cursor Position (RCP)
# \033[999B: Move Down 999 lines (clips at terminal height, effectively moving to the bottom)
# \033[1A: Cursor Up 1, \r: Carriage Return, \033[K: Erase Line

ANSI_SAVE_CURSOR = '\033[s'
ANSI_RESTORE_CURSOR = '\033[u'
# Gets us to the line BEFORE the bottom line (Line L-1).
ANSI_CURSOR_TO_BOTTOM_MINUS_ONE = '\033[999B\033[1A' 

# --- STATUS MANAGEMENT (MODIFIED) ---
status_line_active = False

def delete_last_line( count=1):
    """
    Disables the status line and optionally deletes content lines.
    
    CRITICAL FIX: When using absolute positioning, only perform the relative 
    line deletion (CUU+CR+EL) when clearing temporary content (e.g., user input).
    If count=1, we only unset the flag, because the relative move (\033[1A) 
    was deleting the last line of historical content.
    """
    global status_line_active
    
    if status_line_active:
        assert( count > 0 )
        
        # Only perform the relative line deletion if we are clearing temporary 
        # content (i.e., when count > 1, primarily used in cli_task).
        # If count=1, we skip the destructive deletion.
        if count > 1:
            for i in range(count):
                sys.stdout.write(f"\033[1A\r\033[K") # CUU + CR + EL
            sys.stdout.flush()
            
    # Always set to False when a print block starts or a command is processed.
    status_line_active = False

def print_status_line():
    """
    Prints the status line using absolute positioning (Save/Restore Cursor) 
    to prevent console scrolling and drift, ensuring it always appears at the bottom.
    
    FIXED: Now forces a scroll to preserve log messages before claiming the 
    input line (L-1) and drawing the status line (L).
    """
    global status_line_active
    if not status_line_active: 
        
        stdout = sys.stdout
        
        # 1. Save the current cursor position (P_content_original)
        stdout.write(ANSI_SAVE_CURSOR)
        
        # --- NEW LOG PRESERVATION LOGIC ---
        # 2. Force a scroll: Print a newline and move back up. 
        # This pushes the log message on L-1 into the scroll history (L-2), 
        # making L-1 empty and preserving the log.
        stdout.write('\n\033[1A') 
        # --- END NEW LOG PRESERVATION LOGIC ---

        # 3. Move cursor to the new, absolute input line (Line L-1), which is now empty.
        stdout.write(ANSI_CURSOR_TO_BOTTOM_MINUS_ONE) # \033[999B\033[1A
        
        # 4. We don't need to clear the line here since the scroll in step 2 should have done it.
        # But we will use the clear to be 100% sure before saving.
        stdout.write(f"\r\033[K") 
        
        # 5. Save this now-empty Input Line position (P_input)
        stdout.write(ANSI_SAVE_CURSOR)
        
        # 6. Move cursor DOWN one line to the status line (Line L)
        stdout.write('\033[1B')
        
        # 7. Clear and print the status message on Line L
        caption = f"{status_line_text}> "
        stdout.write(f"\r\033[K{caption}") 
        
        # 8. Restore the cursor to the Input Line position (P_input, Line L-1)
        # This is the empty line where aioconsole.ainput() will start drawing its prompt.
        stdout.write(ANSI_RESTORE_CURSOR)
        
        # 9. Flush the buffer
        stdout.flush()
        
    status_line_active = True



import aioconsole
import asyncio

async def cli_task(stream):
    while True:
        message = await aioconsole.ainput()  # Non-blocking input

        if message:
            # status line shenanigans
            if stream.running:
                delete_last_line(2)
                print(message) # restore the user input we deleted from the console

            # do the commands dance.
            parts = message.split(' ', 1) # Split only on the first space
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else '' # Get args if they exist

            stream.event_callback(stream, "cli_command", (command, args), 2)
            
            if stream.running:
                print_status_line()

        await asyncio.sleep(0.05)