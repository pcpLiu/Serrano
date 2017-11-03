#!/bin/bash
#
# Abort on Error
# This script used to prevent travis-ci terminated due to tons of testing output (4MB limits).
# Credit: Chris Snow(https://stackoverflow.com/questions/26082444/how-to-work-around-travis-cis-4mb-output-limit)
#
set -e

export PING_SLEEP=30s
export WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILD_OUTPUT=$WORKDIR/build.out

touch $BUILD_OUTPUT

stripD() {
    local STRING=${1#$"$2"}
    echo ${STRING%$"$2"}
}


dump_output() {
   echo Tailing the last 500 lines of output:
   tail -500 $BUILD_OUTPUT 
}

# upload_cov() {


#   ############## Upload #######################################################

#   #Gather cov report manually.
#   #
#   #Here we only generate cov report for Mac, sinc iPhone Simulators cannot test Metal realted code.
#   #
#   #Ref: http://mgrebenets.github.io/mobile%20ci/2015/09/21/code-coverage-for-ios-xcode-7

#   # make temp dir
#   mkdir .cov

#   # get building settings
#   BUILD_SETTINGS=.cov/build-settings.txt
#   echo "Gathering report..."
#   xcodebuild -project serrano.xcodeproj \
#     -scheme SerranoFramework \
#     -destination "platform=macOS,arch=x86_64" \
#     -configuration Debug \
#     -showBuildSettings > ${BUILD_SETTINGS}

#   python scripts/coverage_process.py
# }

error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}

error_handler_TESTING() {
  echo ERROR: An error was encountered with the testing.
  dump_output
  #upload_cov
  exit 0
}




############## building #############3
echo "Start building..."

# Set up a repeating loop to send some output to Travis.
bash -c "while true; do echo \$(date) - building ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!

# If an error occurs, run our error handler to output a tail of the build
trap 'error_handler' ERR

echo "Building serrano for macOs"
xcodebuild -project serrano.xcodeproj -scheme SerranoFramework -destination "platform=macOS,arch=x86_64" clean build CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO >> $BUILD_OUTPUT 2>&1

echo "Building MacTestHostingApp for macOs"
xcodebuild -project serrano.xcodeproj -scheme MacTestHostingApp -destination "platform=macOS,arch=x86_64"  clean build  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO  >> $BUILD_OUTPUT 2>&1

echo "Building serrano for iOS"
xcodebuild -project serrano.xcodeproj -scheme SerranoFramework -sdk iphonesimulator -destination "platform=iOS Simulator,name=iPhone 7" clean build >> $BUILD_OUTPUT 2>&1

echo "Building TestingHostApp for iOS"
xcodebuild -project serrano.xcodeproj -scheme TestingHostApp -sdk iphonesimulator -destination "platform=iOS Simulator,name=iPhone 7" clean build >> $BUILD_OUTPUT 2>&1


kill $PING_LOOP_PID
echo "End building"
echo ""
sleep 5

############## testing #######################################################


# disable error handle
set +e
set -e
trap 'error_handler_TESTING' ERR


# echo "Test iOS"
# xcodebuild -project serrano.xcodeproj -scheme SerranoTests -sdk iphonesimulator -destination "platform=iOS Simulator,name=iPhone 7" test  >> $BUILD_OUTPUT 2>&1
# echo "End Test iOS \$(date)"

sleep 5

echo "Test macOS"

bash -c "while true; do echo \$(date) - testing ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!

xcodebuild -project serrano.xcodeproj -scheme SerranoMacTests -destination "platform=macOS,arch=x86_64" test CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO >> $BUILD_OUTPUT 2>&1


echo "End Test macOS"

# The build finished without returning an error so dump a tail of the output
dump_output

#upload_cov()

# nicely terminate the ping output loop
kill $PING_LOOP_PID
