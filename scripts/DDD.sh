 # Project Temp Root ends up with /Build/Intermediates/
  PROJECT_TEMP_ROOT=$(grep -m1 PROJECT_TEMP_ROOT ${BUILD_SETTINGS} | cut -d= -f2 | xargs)
  # xcode 9 path has `.noindex` 
  if [[ "${PROJECT_TEMP_ROOT}" == *".noindex"* ]]; then
    PROJECT_TEMP_ROOT=$(stripD "${PROJECT_TEMP_ROOT}" ".noindex")
  fi
  echo "PROJECT_TEMP_ROOT: ${PROJECT_TEMP_ROOT}"

  # get cov data
  PROFDATA=$(find ${PROJECT_TEMP_ROOT} -name "Coverage.profdata")
  echo "PROFDATA: ${PROFDATA}"

  # find mac app binary
  MAC_BINARY=$(find ${PROJECT_TEMP_ROOT} -path "*Debug/serrano.framework/serrano")
  echo "MAC_BINARY: ${MAC_BINARY}"

  # generate report
  COV_REPORT=.cov/serrano.framework.coverage.txt
  xcrun llvm-cov show \
      -instr-profile ${PROFDATA} \
      ${MAC_BINARY} > ${COV_REPORT}
  echo "COV_REPORT: ${COV_REPORT}"


  # Upload to codcov
  bash <(curl -s https://codecov.io/bash) -f ${COV_REPORT} -t e52f6079-ddb0-4a66-ab7a-95999bef74a4

  # remove temp dir
  rm -r .cov/ 