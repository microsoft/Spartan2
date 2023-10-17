pragma circom 2.0.6;

template cube() {
    signal input x;
    signal input y;
    signal x_sq <== x * x;
    y === x_sq * x;
}

component main { public [x, y] } = cube();