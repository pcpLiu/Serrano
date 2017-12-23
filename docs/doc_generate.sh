#!/bin/bash

tree='2fda3277c6ee4e51b9f05eaa638f70feca901397'

jazzy \
  --clean \
  --author pcpLiu \
  --author_url https://github.com/pcpLiu \
  --github_url https://github.com/pcpLiu/Serrano \
  --github-file-prefix https://github.com/pcpLiu/Serrano/tree/$tree \
  --module-version latest \
  --xcodebuild-arguments -scheme,SerranoFramework \
  --module Serrano \
  --min-acl internal \
  --root-url http://serrano-lib.org/docs/latest/api/ \
  --output generated_docs/api/ \
  --theme fullwidth