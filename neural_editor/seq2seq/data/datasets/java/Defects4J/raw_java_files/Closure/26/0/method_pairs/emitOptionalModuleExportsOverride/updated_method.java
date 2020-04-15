
    private void emitOptionalModuleExportsOverride(Node script,
        String moduleName) {
      if (!modulesWithExports.contains(moduleName)) {
        return;
      }

      Node moduleExportsProp = IR.getprop(IR.name(moduleName),
          IR.string("module$exports"));
      script.addChildToBack(IR.ifNode(
          moduleExportsProp,
          IR.block(IR.exprResult(IR.assign(IR.name(moduleName),
              moduleExportsProp.cloneTree())))).copyInformationFromForTree(
          script));
    }