public void noMoreInteractionsWantedInOrder(Invocation undesired) {
        throw new VerificationInOrderFailure(join(
                "No interactions wanted here:",
                new LocationImpl(),
                "But found this interaction on mock '" + safelyGetMockName(undesired.getMock()) + "':",
                undesired.getLocation()
        ));
    }