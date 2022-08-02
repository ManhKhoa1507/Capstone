#!/bin/sh

# The bin/release script is responsible for providing chaincode metadata to the peer. 
# bin/release is optional. If it is not provided, this step is skipped. 
#
# The peer invokes release with two arguments:
# bin/release BUILD_OUTPUT_DIR RELEASE_OUTPUT_DIR
#
# When release is invoked, BUILD_OUTPUT_DIR contains the artifacts 
# populated by the build program and should be treated as read only input. 
# RELEASE_OUTPUT_DIR is the directory where release must place artifacts to be consumed by the peer.

set -euo pipefail

BUILD_OUTPUT_DIR="$1"
RELEASE_OUTPUT_DIR="$2"

# copy indexes from metadata/* to the output directory
# if [ -d "$BUILD_OUTPUT_DIR/metadata" ] ; then
#    cp -a "$BUILD_OUTPUT_DIR/metadata/"* "$RELEASE_OUTPUT_DIR/"
# fi

#external chaincodes expect artifacts to be placed under "$RELEASE_OUTPUT_DIR"/chaincode/server
if [ -f $BUILD_OUTPUT_DIR/connection.json ]; then
   mkdir -p "$RELEASE_OUTPUT_DIR"/chaincode/server
   cp $BUILD_OUTPUT_DIR/connection.json "$RELEASE_OUTPUT_DIR"/chaincode/server

   #if tls_required is true, copy TLS files (using above example, the fully qualified path for these fils would be "$RELEASE_OUTPUT_DIR"/chaincode/server/tls)

   exit 0
fi

exit 1