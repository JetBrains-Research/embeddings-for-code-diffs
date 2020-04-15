
    private void emitOptionalModuleExportsOverride(Node script,
        String moduleName) {

      Node moduleExportsProp = IR.getprop(IR.name(moduleName),
          IR.string("module$exports"));
      script.addChildToBack(IR.ifNode(
          moduleExportsProp,
          IR.block(IR.exprResult(IR.assign(IR.name(moduleName),
              moduleExportsProp.cloneTree())))).copyInformationFromForTree(
          script));
    }