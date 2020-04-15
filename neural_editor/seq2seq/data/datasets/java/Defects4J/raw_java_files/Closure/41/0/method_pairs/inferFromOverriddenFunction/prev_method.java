
  FunctionTypeBuilder inferFromOverriddenFunction(
      @Nullable FunctionType oldType, @Nullable Node paramsParent) {
    if (oldType == null) {
      return this;
    }

    returnType = oldType.getReturnType();
    returnTypeInferred = oldType.isReturnTypeInferred();
    if (paramsParent == null) {
      // Not a function literal.
      parametersNode = oldType.getParametersNode();
      if (parametersNode == null) {
        parametersNode = new FunctionParamBuilder(typeRegistry).build();
      }
    } else {
      // We're overriding with a function literal. Apply type information
      // to each parameter of the literal.
      FunctionParamBuilder paramBuilder =
          new FunctionParamBuilder(typeRegistry);
      Iterator<Node> oldParams = oldType.getParameters().iterator();
      boolean warnedAboutArgList = false;
      boolean oldParamsListHitOptArgs = false;
      for (Node currentParam = paramsParent.getFirstChild();
           currentParam != null; currentParam = currentParam.getNext()) {
        if (oldParams.hasNext()) {
          Node oldParam = oldParams.next();
          Node newParam = paramBuilder.newParameterFromNode(oldParam);

          oldParamsListHitOptArgs = oldParamsListHitOptArgs ||
              oldParam.isVarArgs() ||
              oldParam.isOptionalArg();

          // The subclass method might write its var_args as individual
          // arguments.
          if (currentParam.getNext() != null && newParam.isVarArgs()) {
            newParam.setVarArgs(false);
            newParam.setOptionalArg(true);
          }
        } else {
          warnedAboutArgList |= addParameter(
              paramBuilder,
              typeRegistry.getNativeType(UNKNOWN_TYPE),
              warnedAboutArgList,
              codingConvention.isOptionalParameter(currentParam) ||
                  oldParamsListHitOptArgs,
              codingConvention.isVarArgsParameter(currentParam));
        }
      }

      // Clone any remaining params that aren't in the function literal.

      parametersNode = paramBuilder.build();
    }
    return this;
  }