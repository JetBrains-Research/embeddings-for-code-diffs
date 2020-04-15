
    public static LocalDate fromCalendarFields(Calendar calendar) {
        if (calendar == null) {
            throw new IllegalArgumentException("The calendar must not be null");
        }
        int era = calendar.get(Calendar.ERA);
        int yearOfEra = calendar.get(Calendar.YEAR);
        return new LocalDate(
            (era == GregorianCalendar.AD ? yearOfEra : 1 - yearOfEra),
            calendar.get(Calendar.MONTH) + 1,
            calendar.get(Calendar.DAY_OF_MONTH)
        );
    }