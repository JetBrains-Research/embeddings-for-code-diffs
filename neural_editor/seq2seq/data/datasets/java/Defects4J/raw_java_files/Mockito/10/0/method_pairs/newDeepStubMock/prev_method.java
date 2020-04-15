
    private Object newDeepStubMock(GenericMetadataSupport returnTypeGenericMetadata) {
        return mockitoCore().mock(
                returnTypeGenericMetadata.rawType(),
                withSettingsUsing(returnTypeGenericMetadata)
        );
    }