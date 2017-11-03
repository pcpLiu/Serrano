Pod::Spec.new do |spec|
  spec.name         = 'Serrano'
  spec.version      = '0.1.1-alpha'
  spec.license      = { :type => 'MIT' }
  spec.homepage     = 'https://github.com/pcpLiu/Serrano'
  spec.authors      = { 'Tim Liu' => 'pcpliu.dev@gmail.com' }
  spec.summary      = 'Graph computation library for iOS'
  spec.source       = { :git => 'https://github.com/pcpLiu/Serrano.git', :tag => 'v0.1.1-alpha' }
  spec.module_name  = 'Serrano'
  spec.documentation_url = 'http://serrano-lib.org'

  spec.ios.deployment_target  = '10.0'
  spec.osx.deployment_target  = '10.11'
  spec.requires_arc = true

  spec.framework = 'Accelerate'
  spec.weak_framework = 'Metal'

  spec.source_files = 'Source/**/*.{swift,c,h}'
  spec.public_header_files = 'Source/SupportingFiles/Serrano.h'
  spec.resources = 'Source/**/*.{metal}'

  spec.pod_target_xcconfig = { 'SWIFT_VERSION' => '3.2', 'SWIFT_INCLUDE_PATHS' => '$(SRCROOT)/Serrano/Source/library/FBSUtil/**'}
  spec.preserve_paths = 'Source/library/FBSUtil/module.modulemap'

end