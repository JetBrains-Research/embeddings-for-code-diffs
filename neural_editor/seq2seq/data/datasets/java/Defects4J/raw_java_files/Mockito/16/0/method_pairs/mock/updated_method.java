
    public static <T> T mock(Class<T> classToMock, MockSettings mockSettings) {
        return MOCKITO_CORE.mock(classToMock, mockSettings, true);
    }