function y = cliff(x1, x2)
    x1 = (2 * x1 - 1) * 20;
    x2 = (2 * x2 - 1) * 7.5 - 2.5;
    term1 = -0.5 * x1 .^ 2 / 100;
    term2 = -0.5 * (x2 + 0.03 * x1 .^ 2 - 3) .^ 2;
    y = 5 * exp(term1 + term2);
end