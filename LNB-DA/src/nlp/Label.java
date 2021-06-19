package nlp;

public class Label {
	// private static final String[] labelStrs = { "POS", "NEG" };

	public static String convertPostiveNegativeToPlusOneMinusOne(String labelStr) {
		if (labelStr.equals("POS") || labelStr.equals("+1")) {
			return "+1";
		} else {
			return "-1";
		}
	}

	public static String convertPlusOneMinusOneToPostiveNegative(String labelStr) {
		if (labelStr.equals("+1") || labelStr.equals("POS")) {
			return "POS";
		} else {
			return "NEG";
		}
	}

	public static int convertLabelStrToLabelInteger(String labelStr) {
		if (isPositive(labelStr)) {
			return 1;
		} else {
			return 0;
		}
	}

	public static String convertLabelIntegertoLabelStr(int labelInteger) {
		if (labelInteger == 1) {
			return "+1";
		} else {
			return "-1";
		}
	}

	public static boolean isPositive(String label) {
		return label.equals("POS") || label.equals("+1");
	}

	public static boolean isNegative(String label) {
		return label.equals("NEG") || label.equals("-1");
	}

	// public static int getLabel(String labelStr) {
	// for (int i = 0; i < labelStrs.length; ++i) {
	// if (labelStrs[i].equals(labelStr)) {
	// return i;
	// }
	// }
	// ExceptionUtility.throwAndCatchException(labelStr
	// + " is not recognizable!");
	// return -1;
	// }
}
