# Test on local real iOS device 
# usage: (project root dir)$ ./scripts/test_real_device.sh [device UDID]


EXPECT_TIME="1 hour"
WARNING="\n\n*****Note: Tests are expected to running around ${EXPECT_TIME} [Depends on your device]*****\n\n"
echo -e "\033[0;31m${WARNING}\x1b[m"
sleep 5

#### Build and test

DEVICE_UDID="$1"

xcodebuild -project serrano.xcodeproj -scheme serrano -destination "platform=iOS,id=${DEVICE_UDID}" clean build

xcodebuild -project serrano.xcodeproj -scheme TestingHostApp -destination "platform=iOS,id=${DEVICE_UDID}" clean build

xcodebuild -project serrano.xcodeproj -scheme serrano -destination "platform=iOS,id=${DEVICE_UDID}"  test 
