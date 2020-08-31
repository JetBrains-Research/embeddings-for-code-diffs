
    private Object createNewDeepStubMock(GenericMetadataSupport returnTypeGenericMetadata) {
        return mock(
                returnTypeGenericMetadata.rawType(),
                withSettingsUsing(returnTypeGenericMetadata)
        );
    }