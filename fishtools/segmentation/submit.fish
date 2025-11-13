#!/usr/bin/env fish

# --- Configuration ---
# Removed: set LOCAL_SOURCE_DIR "/path/to/local/source/folder"
set REMOTE_HOST "chaichontat@sob"
set PYTHON_COMMAND "conda activate torch && python /fast2/fishtools/useful/distributed/distributed_segmentation.py /fast2/cellpose/data"
set REMOTE_DEST_DIR "/fast2/cellpose/data/"
# Construct the command to run inside tmux and then close the pane
set REMOTE_TMUX_COMMAND "tmux new-window '$PYTHON_COMMAND ; tmux kill-pane'"

# --- Argument Parsing ---
set -l options (fish_opt --short=d --long=delete)
argparse $options -- $argv
if set --query --local _flag_d # Check if -d or --delete was passed
    set DELETE_REMOTE true
else
    set DELETE_REMOTE false
end

# Check for the required positional argument (source directory)
if test (count $argv) -ne 1
    echo "Usage: $argv[0] [-d|--delete] <local_source_directory>" >&2
    echo "  -d, --delete : Delete the remote destination directory before syncing." >&2
    exit 1
end

set LOCAL_SOURCE_DIR $argv[1]

# Check if the provided source directory exists
if not test -d "$LOCAL_SOURCE_DIR"
    echo "Error: Local source directory '$LOCAL_SOURCE_DIR' not found." >&2
    exit 1
end



# --- Optional Remote Deletion ---
if $DELETE_REMOTE
    echo "Step 0: Deleting remote destination directory '$REMOTE_DEST_DIR'..."
    # Pass 'rm -rf' and the directory as separate arguments to ssh
    if ssh "$REMOTE_HOST" rm -rf "$REMOTE_DEST_DIR"
        echo "Remote directory deleted successfully."
    else
        # Added error message and exit
        echo "Error: Failed to delete remote directory '$REMOTE_DEST_DIR'." >&2
    end
end


echo "\nStep 1: Rsyncing source data from '$LOCAL_SOURCE_DIR' to remote host..."
if rsync -arv --info=progress2 --info=name0 --delete "$LOCAL_SOURCE_DIR/" "$REMOTE_HOST":"$REMOTE_DEST_DIR"
    echo "Rsync to remote host successful."
else
    echo "Error: Rsync to remote host failed." >&2
    exit 1
end

echo -e "\nStep 2: Running Python command on remote host inside tmux..."

# Use the tmux command in the ssh call
if ssh "$REMOTE_HOST" "$REMOTE_TMUX_COMMAND"
    echo "Remote Python command executed successfully in tmux."
else
    echo "Error: Remote Python command (in tmux) failed." >&2
    exit 1
end

echo -e "\nStep 3: Rsyncing results back from remote host..."
# Ensure the local results directory exists
# mkdir -p "$LOCAL_RESULTS_DIR"
# if rsync -avz "$REMOTE_HOST":"$REMOTE_DEST_DIR" "$LOCAL_SOURCE_DIR"
#     echo "Rsync from remote host successful."
# else
#     echo "Error: Rsync from remote host failed." >&2
#     exit 1
# end

echo -e "\nProcess completed successfully."
exit 0