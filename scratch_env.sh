# scratch_env.sh — keep every library cache, config and temp file OFF the home directory.
#   source scratch_env.sh          (run_all.sh does this automatically)
# Override the location with QVR_SCRATCH_HOME before sourcing.
export QVR_SCRATCH_HOME=${QVR_SCRATCH_HOME:-/scratch90/chris_/quantum/.home}
export XDG_CACHE_HOME=$QVR_SCRATCH_HOME/cache        # astropy/gwpy, generic caches
export XDG_CONFIG_HOME=$QVR_SCRATCH_HOME/config      # astropy config
export XDG_DATA_HOME=$QVR_SCRATCH_HOME/data
export MPLCONFIGDIR=$QVR_SCRATCH_HOME/matplotlib     # matplotlib font cache
export PIP_CACHE_DIR=$QVR_SCRATCH_HOME/pip           # pip downloads
export TMPDIR=$QVR_SCRATCH_HOME/tmp                  # temporary files (downloads, tempfile)
# astropy only uses the XDG locations if its subfolders already exist
mkdir -p "$XDG_CACHE_HOME/astropy" "$XDG_CONFIG_HOME/astropy" "$XDG_DATA_HOME" \
         "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$TMPDIR"
