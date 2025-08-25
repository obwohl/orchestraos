// RUN: not %orchestra-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

// CHECK: error: target attribute requires 'arch' attribute of type StringAttr
orchestra.schedule {
  orchestra.task target = #orchestra.target<device_id = 0> {
    orchestra.return
  }
}

// -----

// CHECK: error: target attribute requires 'device_id' attribute of type IntegerAttr
orchestra.schedule {
  orchestra.task target = #orchestra.target<arch = "gpu"> {
    orchestra.return
  }
}
