#!/bin/bash

jazzy \
  --clean \
  --author pcpLiu \
  --author_url https://github.com/pcpLiu \
  --github_url https://github.com/pcpLiu/Serrano \
  --github-file-prefix https://github.com/pcpLiu/Serrano/tree/v0.1.5-alpha \
  --module-version v0.1.5-alpha \
  --xcodebuild-arguments -scheme,SerranoFramework \
  --module Serrano \
  --min-acl internal \
  --root-url http://serrano-lib.org/docs/v0.1.5-alpha/api/ \
  --output generated_docs/v0.1.5-alpha/api/ \
  --theme fullwidth