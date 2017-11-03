#!/bin/bash

jazzy \
  --clean \
  --author pcpLiu \
  --author_url https://github.com/pcpLiu \
  --github_url https://github.com/pcpLiu/Serrano \
  --github-file-prefix https://github.com/pcpLiu/Serrano/tree/v0.1.0-alpha \
  --module-version v0.1.0-alpha \
  --xcodebuild-arguments -scheme,SerranoFramework \
  --module Serrano \
  --root-url http://serrano-lib.org/docs/v0.1.0-alpha/api/ \
  --output docs/v0.1.0-alpha/api/ \
  --theme fullwidth