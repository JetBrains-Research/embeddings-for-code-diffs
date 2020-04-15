
    private Object newDeepStubMock(GenericMetadataSupport returnTypeGenericMetadata, Object parentMock) {
        MockCreationSettings parentMockSettings = new MockUtil().getMockSettings(parentMock);
        return mockitoCore().mock(
                returnTypeGenericMetadata.rawType(),
                withSettingsUsing(returnTypeGenericMetadata, parentMockSettings)
        );
    }