public Object answer(InvocationOnMock invocation) throws Throwable {
    	if (Modifier.isAbstract(invocation.getMethod().getModifiers())) {
    		return new GloballyConfiguredAnswer().answer(invocation);
    	}
        return invocation.callRealMethod();
    }