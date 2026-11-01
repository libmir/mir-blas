/++
Low-level ndslice wrapper for BLAS.

Vector arguments follow the logical order of their ndslice views, including
reversed views and non-unit strides. Maximum indices refer to that logical
order; for inputs without NaNs, ties select the first maximum. Maximum-index
results for NaN-containing inputs are unspecified and may depend on the backend
and view layout.

Authors: Ilya Yaroshenko
Copyright:  Copyright © 2017, Symmetry Investments & Kaleidic Associates
+/
module mir.blas;

import mir.ndslice.dynamic;
import mir.ndslice.iterator: StairsIterator;
import mir.ndslice.slice;
import mir.ndslice.topology;

import std.traits: isFloatingPoint;

static import cblas;
public import cblas: Uplo, Side;

@safe pure nothrow @nogc
private auto matrixStride(S)(S a)
 if (S.N == 2)
{
    assert(a._stride!1 == 1);
    return a._stride != 1 ? a._stride : a.length!1;
}

// ndslice points at the first logical element. BLAS applies its own starting
// offset for negative increments, so supply the lowest-addressed element.
// For a nonempty valid view this is another element of the same allocation;
// empty views must not perform pointer arithmetic. Used only by trusted FFI calls.
pragma(inline, true)
private auto vectorBase(T, SliceKind kind)(Slice!(T*, 1, kind) x)
    @system pure nothrow @nogc
{
    return x._stride < 0 && x.length
        ? x.iterator + cast(ptrdiff_t)(x.length - 1) * x._stride
        : x.iterator;
}

// These order-independent operations require positive native increments.
private auto forwardVector(T, SliceKind kind)(Slice!(T*, 1, kind) x)
    @safe pure nothrow @nogc
{
    static if (kind == Universal)
        if (x._stride < 0)
            return x.reversed;
    return x;
}

///
@trusted pure nothrow @nogc
T dot(T,
    SliceKind kindX,
    SliceKind kindY,
    )(
    Slice!(const(T)*, 1, kindX) x,
    Slice!(const(T)*, 1, kindY) y,
    )
in
{
    assert(x.length == y.length);
}
do
{
    return cblas.dot(
        cast(cblas.blasint) x.length,

        vectorBase(x),
        cast(cblas.blasint) x._stride,

        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

///
@trusted pure nothrow @nogc
auto nrm2(T,
    SliceKind kindX,
    )(
    Slice!(const(T)*, 1, kindX) x,
    )
{
    x = forwardVector(x);
    return cblas.nrm2(
        cast(cblas.blasint) x.length,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

///
@trusted pure nothrow @nogc
auto asum(T,
    SliceKind kindX,
    )(
    Slice!(const(T)*, 1, kindX) x,
    )
{
    x = forwardVector(x);
    return cblas.asum(
        cast(cblas.blasint) x.length,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

///
@trusted pure nothrow @nogc
void axpy(T,
    SliceKind kindX,
    SliceKind kindY,
    )(
    T a,
    Slice!(const(T)*, 1, kindX) x,
    Slice!(T*, 1, kindY) y,
    )
in
{
    assert(x.length == y.length);
}
do
{
    cblas.axpy(
        cast(cblas.blasint) x.length,
        a,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

///
@trusted pure nothrow @nogc
void scal(T,
    SliceKind kindX,
    )(
    T a,
    Slice!(T*, 1, kindX) x,
    )
{
    x = forwardVector(x);
    cblas.scal(
        cast(cblas.blasint) x.length,
        a,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

///
@trusted pure nothrow @nogc
void copy(T,
    SliceKind kindX,
    SliceKind kindY,
    )(
    Slice!(const(T)*, 1, kindX) x,
    Slice!(T*, 1, kindY) y,
    )
in
{
    assert(x.length == y.length);
}
do
{
    cblas.copy(
        cast(cblas.blasint) x.length,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

///
@trusted pure nothrow @nogc
void swap(T,
    SliceKind kindX,
    SliceKind kindY,
    )(
    Slice!(T*, 1, kindX) x,
    Slice!(T*, 1, kindY) y,
    )
in
{
    assert(x.length == y.length);
}
do
{
    cblas.swap(
        cast(cblas.blasint) x.length,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

///
@trusted pure nothrow @nogc
void ger(T,
    SliceKind kindX,
    SliceKind kindY,
    SliceKind kindA,
    )(
    T alpha,
    Slice!(const(T)*, 1, kindX) x,
    Slice!(const(T)*, 1, kindY) y,
    Slice!(T*, 2, kindA) a,
    )
in
{
    assert(a.length!0 == x.length);
    assert(a.length!1 == y.length);
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    static if (isFloatingPoint!T)
        alias gerImpl = cblas.ger;
    else
        alias gerImpl = cblas.geru;
    gerImpl(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,

        cast(cblas.blasint) x.length,
        cast(cblas.blasint) y.length,

        alpha,

        vectorBase(x),
        cast(cblas.blasint) x._stride,

        vectorBase(y),
        cast(cblas.blasint) y._stride,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,
    );
}

///
@trusted pure nothrow @nogc
void gerc(T,
    SliceKind kindX,
    SliceKind kindY,
    SliceKind kindA,
    )(
    T alpha,
    Slice!(const(T)*, 1, kindX) x,
    Slice!(const(T)*, 1, kindY) y,
    Slice!(T*, 2, kindA) a,
    )
in
{
    assert(a.length!0 == x.length);
    assert(a.length!1 == y.length);
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    static if (isFloatingPoint!T)
        alias gerImpl = cblas.ger;
    else
        alias gerImpl = cblas.gerc;
    gerImpl(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,

        cast(cblas.blasint) x.length,
        cast(cblas.blasint) y.length,

        alpha,

        vectorBase(x),
        cast(cblas.blasint) x._stride,

        vectorBase(y),
        cast(cblas.blasint) y._stride,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,
    );
}

version(unittest)
{
    /// Construct real or complex test values with old and new CBLAS bindings.
    // cblas 2.0.4 through 2.1.1 default to built-in complex types, which do not
    // support the struct-style constructors used by newer bindings. Remove this
    // helper and replace its calls with direct constructors only after raising
    // the minimum cblas version to 2.2.0 or later and checking the tests with
    // every supported complex configuration, including CBLAS_STD_COMPLEX.
    // This compatibility requirement is independent of the DMD #24119 workaround.
    private T testValue(T)(real re, real im = 0) @safe pure nothrow @nogc
    {
        static if (isFloatingPoint!T)
        {
            assert(im == 0);
            return cast(T) re;
        }
        else static if (is(T == cfloat) || is(T == cdouble) || is(T == creal))
            return cast(T) (re + im * 1.0Li);
        else
            return T(cast(typeof(T.init.re)) re, cast(typeof(T.init.im)) im);
    }
}

// Rectangular updates keep logical dimensions when storage is transposed.
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(float, double, cblas._cfloat, cblas._cdouble))
    {{
        foreach (m; [2, 3])
        foreach (transposeA; [false, true])
        foreach (conjugate; [false, true])
        {
            auto n = 5 - m;
            T[20] storage, reference;
            storage[] = reference[] = testValue!T(-99);
            auto a = storage[].sliced(4, 5)
                [0 .. (transposeA ? n : m), 0 .. (transposeA ? m : n)].universal;
            auto expected = reference[].sliced(4, 5)
                [0 .. (transposeA ? n : m), 0 .. (transposeA ? m : n)].universal;
            if (transposeA)
            {
                a = a.transposed;
                expected = expected.transposed;
            }
            T[3] xs, ys;
            foreach (i; 0 .. m) xs[i] = testValue!T(i + 1);
            foreach (j; 0 .. n)
            {
                static if (isFloatingPoint!T) ys[j] = testValue!T(j + 2);
                else ys[j] = testValue!T(j + 2, j + 1);
            }
            foreach (i; 0 .. m)
            foreach (j; 0 .. n)
            {
                auto y = ys[j];
                static if (!isFloatingPoint!T)
                    if (conjugate) y = testValue!T(y.re, -y.im);
                a[i, j] = testValue!T(1);
                expected[i, j] = testValue!T(1) + testValue!T(2) * xs[i] * y;
            }
            if (conjugate)
                gerc(testValue!T(2), xs[0 .. m].sliced.lightConst, ys[0 .. n].sliced.lightConst, a);
            else
                ger(testValue!T(2), xs[0 .. m].sliced.lightConst, ys[0 .. n].sliced.lightConst, a);
            // Also verify that padding outside the supplied view is untouched.
            assert(storage == reference);
        }
    }}
}

// Rectangular updates also support reversed, nonunit complex vectors.
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    static foreach (T; AliasSeq!(cblas._cfloat, cblas._cdouble))
    {{
        foreach (m; [2, 3])
        foreach (transposeA; [false, true])
        foreach (conjugate; [false, true])
        foreach (reverseX; [false, true])
        foreach (reverseY; [false, true])
        {
            auto n = 5 - m;
            T[20] storage, reference;
            storage[] = reference[] = testValue!T(-99);
            auto a = storage[].sliced(4, 5)
                [0 .. (transposeA ? n : m), 0 .. (transposeA ? m : n)].universal;
            auto expected = reference[].sliced(4, 5)
                [0 .. (transposeA ? n : m), 0 .. (transposeA ? m : n)].universal;
            if (transposeA)
            {
                a = a.transposed;
                expected = expected.transposed;
            }
            T[8] xStorage, yStorage;
            xStorage[] = yStorage[] = testValue!T(-99);
            auto xs = xStorage[1 .. 2 * m].sliced.universal.strided(2);
            auto ys = yStorage[1 .. 2 * n].sliced.universal.strided(2);
            if (reverseX) xs = xs.reversed;
            if (reverseY) ys = ys.reversed;
            static if (isFloatingPoint!T) auto alpha = testValue!T(2);
            else auto alpha = testValue!T(2, -1);
            foreach (i; 0 .. m)
            {
                static if (isFloatingPoint!T) xs[i] = testValue!T(i + 1);
                else xs[i] = testValue!T(i + 1, i + 2);
            }
            foreach (j; 0 .. n)
            {
                static if (isFloatingPoint!T) ys[j] = testValue!T(j + 2);
                else ys[j] = testValue!T(j + 2, j + 1);
            }
            foreach (i; 0 .. m)
            foreach (j; 0 .. n)
            {
                auto y = ys[j];
                static if (!isFloatingPoint!T)
                    if (conjugate) y = testValue!T(y.re, -y.im);
                a[i, j] = testValue!T(1);
                expected[i, j] = testValue!T(1) + alpha * xs[i] * y;
            }
            if (conjugate)
                gerc(alpha, xs.lightConst, ys.lightConst, a);
            else
                ger(alpha, xs.lightConst, ys.lightConst, a);
            // Also verify that padding outside the supplied view is untouched.
            assert(storage == reference);
            foreach (i; 0 .. 8)
            {
                if (i % 2 == 0 || i >= 2 * m) assert(xStorage[i] == testValue!T(-99));
                if (i % 2 == 0 || i >= 2 * n) assert(yStorage[i] == testValue!T(-99));
            }
        }
    }}
}

///
@trusted pure nothrow @nogc
void gemv(T,
    SliceKind kindA,
    SliceKind kindX,
    SliceKind kindY,
    )(
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindX) x,
    T beta,
    Slice!(T*, 1, kindY) y,
    )
in
{
    assert(a.length!1 == x.length);
    assert(a.length!0 == y.length);
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    cblas.gemv(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,
        cblas.Transpose.NoTrans,
        
        cast(cblas.blasint) y.length,
        cast(cblas.blasint) x.length,

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        vectorBase(x),
        cast(cblas.blasint) x._stride,

        beta,

        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

///
@trusted pure nothrow @nogc
void gemm(T,
    SliceKind kindA,
    SliceKind kindB,
    SliceKind kindC,
    )(
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b,
    T beta,
    Slice!(T*, 2, kindC) c,
    )
in
{
    assert(a.length!1 == b.length!0);
    assert(a.length!0 == c.length!0);
    assert(c.length!1 == b.length!1);
}
do
{
    auto k = cast(cblas.blasint) a.length!1;

    static if (kindC == Universal)
    {
        if (c._stride!1 != 1)
        {
            assert(c._stride!0 == 1, "Matrix C must have a stride equal to 1.");
            .gemm(
                alpha,
                b.universal.transposed,
                a.universal.transposed,
                beta,
                c.transposed.assumeCanonical);
            return;
        }
        assert(c._stride!1 == 1, "Matrix C must have a stride equal to 1.");
    }
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    static if (kindB == Universal)
    {
        bool transB;
        if (b._stride!1 != 1)
        {
            b = b.transposed;
            transB = true;
        }
        assert(b._stride!1 == 1, "Matrix B must have a stride equal to 1.");
    }
    else
        enum transB = false;

    cblas.gemm(
        cblas.Order.RowMajor,
        transA ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,
        transB ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,
        
        cast(cblas.blasint) c.length!0,
        cast(cblas.blasint) c.length!1, 
        cast(cblas.blasint) k,

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        b.iterator,
        cast(cblas.blasint) b.matrixStride,

        beta,

        c.iterator,
        cast(cblas.blasint) c.matrixStride,
    );
}

///
@trusted pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto a = [3.0, 5, 2, 4, 2, 3].sliced(2, 3).universal;
    auto b = [2.0, 3, 4].sliced(3, 1).universal;

    auto c = [100.0, 100].sliced(2, 1).universal;
    gemm(1.0, a, b, 1.0, c);
    assert(c == [[6 + 15 + 8 + 100], [8 + 6 + 12 + 100]]);
}

@trusted pure nothrow
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;
    import mir.ndslice.dynamic: transposed;

    auto a = [3.0, 5, 2, 4, 2, 3].sliced(2, 3).universal;
    auto b = [2.0, 3, 4].sliced(3, 1).universal;

    auto c = [100.0, 100].sliced(1, 2).transposed.universal;
    gemm(1.0, a, b, 1.0, c);
    assert(c == [[6 + 15 + 8 + 100], [8 + 6 + 12 + 100]]);
}

///
@trusted pure nothrow @nogc
void syrk(T,
    SliceKind kindA,
    SliceKind kindC,
    )(
    Uplo uplo,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    T beta,
    Slice!(T*, 2, kindC) c,
    )
in
{
    assert(a.length!0 == c.length!0);
    assert(c.length!1 == c.length!0);
}
do
{
    auto k = a.length!1;
    static if (kindC == Universal)
    {
        if (c._stride!1 != 1)
        {
            c = c.transposed;
            uplo = uplo == Uplo.Lower ? Uplo.Upper : Uplo.Lower;
        }
        assert(c._stride!1 == 1, "Matrix C must have a stride equal to 1.");
    }
    else
        enum transC = false;
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    cblas.syrk(
        cblas.Order.RowMajor,
        uplo,
        transA ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,

        cast(cblas.blasint) c.length!0,
        cast(cblas.blasint) k,

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        beta,

        c.iterator,
        cast(cblas.blasint) c._stride,
    );
}

///
@trusted pure nothrow @nogc
void trmm(T,
    SliceKind kindA,
    SliceKind kindB,
    )(
    cblas.Side side,
    cblas.Uplo uplo,
    cblas.Diag diag,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(T*, 2, kindB) b,
    )
in
{
    assert(a.length!1 == a.length!0);
    assert(a.length == (side == cblas.Side.Left ? b.length!0 : b.length!1));
}
do
{
    static if (kindB == Universal)
    {
        if (b._stride!1 != 1)
        {
            assert(b._stride!0 == 1, "Matrix B must have a stride equal to 1.");
            return .trmm(
                side == cblas.Side.Left ? cblas.Side.Right : cblas.Side.Left,
                uplo == cblas.Uplo.Upper ? cblas.Uplo.Lower : cblas.Uplo.Upper,
                diag,
                alpha,
                a.transposed,
                b.transposed.assumeCanonical,
            );
        }
    }

    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            // UPLO describes the stored matrix before applying TransA.
            uplo = uplo == cblas.Uplo.Upper ? cblas.Uplo.Lower : cblas.Uplo.Upper;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;

    cblas.trmm(
        cblas.Order.RowMajor,
        side,
        uplo,
        transA ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,
        diag,

        cast(cblas.blasint) b.length!0,
        cast(cblas.blasint) b.length!1,

        alpha,

        a.iterator,
        cast(cblas.blasint) a._stride,

        b.iterator,
        cast(cblas.blasint) b.matrixStride,
    );
}

// Exercise logical triangles independently of their physical storage layout.
@safe pure nothrow @nogc
unittest
{
    import std.meta: AliasSeq;
    import std.math: fabs;
    static foreach (T; AliasSeq!(double, cblas._cfloat, cblas._cdouble))
    {{
        foreach (side; [cblas.Side.Left, cblas.Side.Right])
        foreach (uplo; [cblas.Uplo.Upper, cblas.Uplo.Lower])
        foreach (diag; [cblas.Diag.NonUnit, cblas.Diag.Unit])
        foreach (transposeA; [false, true])
        foreach (transposeB; [false, true])
        {
            // Padded leading dimensions catch accidental assumptions of contiguity.
            T[15] storageA;
            T[15] storageB;
            storageB[] = testValue!T(-99);
            static if (isFloatingPoint!T) auto alpha = testValue!T(2);
            else auto alpha = testValue!T(2, 1);
            auto a = storageA[].sliced(3, 5)[0 .. 3, 0 .. 3].universal;
            if (transposeA)
                a = a.transposed;
            size_t rows = side == cblas.Side.Left ? 3 : 2;
            size_t cols = side == cblas.Side.Left ? 2 : 3;
            auto b = storageB[].sliced(3, 5)
                [0 .. (transposeB ? cols : rows), 0 .. (transposeB ? rows : cols)].universal;
            if (transposeB)
                b = b.transposed;
            T[9] dense;
            foreach (i; 0 .. 3)
            foreach (j; 0 .. 3)
            {
                bool selected = uplo == cblas.Uplo.Upper ? i <= j : i >= j;
                static if (isFloatingPoint!T)
                    auto value = testValue!T(i == j ? 2 : i + j + 1);
                else
                    auto value = testValue!T(i == j ? 2 : i + j + 1, i == j ? 1 : i + 1);
                dense[i * 3 + j] = selected ? value : testValue!T(0);
                if (i == j && diag == cblas.Diag.Unit)
                    dense[i * 3 + j] = testValue!T(1);
                // Keep separate assignments to avoid a Windows x64 DMD -inline crash
                // with built-in cfloat: https://github.com/dlang/dmd/issues/24119
                if (selected && !(i == j && diag == cblas.Diag.Unit))
                    a[i, j] = value;
                else
                    a[i, j] = testValue!T(double.nan);
            }
            T[6] original;
            T[6] expected;
            foreach (i; 0 .. rows)
            foreach (j; 0 .. cols)
            {
                static if (isFloatingPoint!T) auto value = testValue!T(i * cols + j + 1);
                else auto value = testValue!T(i * cols + j + 1, j + 1);
                original[i * cols + j] = b[i, j] = value;
            }
            foreach (i; 0 .. rows)
            foreach (j; 0 .. cols)
            {
                T sum = testValue!T(0);
                foreach (k; 0 .. 3)
                    sum += side == cblas.Side.Left
                        ? dense[i * 3 + k] * original[k * cols + j]
                        : original[i * cols + k] * dense[k * 3 + j];
                expected[i * cols + j] = alpha * sum;
            }
            trmm(side, uplo, diag, alpha, a.lightConst, b);
            foreach (i; 0 .. rows)
            foreach (j; 0 .. cols)
            {
                assert(b[i, j] == expected[i * cols + j]);
                // Supply the independent reference to the solve regression.
                b[i, j] = expected[i * cols + j];
            }
            foreach (i; 0 .. 3)
            foreach (j; 0 .. 5)
                if (i >= (transposeB ? cols : rows) || j >= (transposeB ? rows : cols))
                    assert(storageB[i * 5 + j] == testValue!T(-99));
            trsm(side, uplo, diag, testValue!T(1) / alpha, a.lightConst, b);
            foreach (i; 0 .. rows)
            foreach (j; 0 .. cols)
            {
                auto difference = b[i, j] - original[i * cols + j];
                static if (is(T == cblas._cfloat)) enum tolerance = 1e-4;
                else enum tolerance = 1e-12;
                static if (isFloatingPoint!T) assert(fabs(difference) < tolerance);
                else assert(fabs(difference.re) < tolerance && fabs(difference.im) < tolerance);
            }
            foreach (i; 0 .. 3)
            foreach (j; 0 .. 5)
                if (i >= (transposeB ? cols : rows) || j >= (transposeB ? rows : cols))
                    assert(storageB[i * 5 + j] == testValue!T(-99));

        }
    }}
}

/// Example: Left/Upper/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [29.0, 18.0, 4.0].sliced(3, 1);

    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3).universal;
    auto b = [2.0, 3, 4].sliced(3, 1).universal;
    trmm(cblas.Side.Left, cblas.Uplo.Upper, cblas.Diag.NonUnit, 1.0, a, b);
    assert(b.equal(result));
}

/// Example: Right/Upper/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [6.0, 16, 17].sliced(1, 3);

    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3).universal;
    auto b = [2.0, 3, 4].sliced(1, 3).universal;
    trmm(cblas.Side.Right, cblas.Uplo.Upper, cblas.Diag.NonUnit, 1.0, a, b);
    assert(b.equal(result));
}

/// Example: Left/Lower/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [6.0, 16, 17].sliced(3, 1);

    auto a = [3.0, 0, 0, 5, 2, 0, 2, 3, 1].sliced(3, 3).universal;
    auto b = [2.0, 3, 4].sliced(3, 1).universal;
    trmm(cblas.Side.Left, cblas.Uplo.Lower, cblas.Diag.NonUnit, 1.0, a, b);
    assert(b.equal(result));
}

/// Example: Right/Lower/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [29.0, 18.0, 4.0].sliced(1, 3);

    auto a = [3.0, 0, 0, 5, 2, 0, 2, 3, 1].sliced(3, 3).universal;
    auto b = [2.0, 3, 4].sliced(1, 3).universal;
    trmm(cblas.Side.Right, cblas.Uplo.Lower, cblas.Diag.NonUnit, 1.0, a, b);
    assert(b.equal(result));
}

/// Example: Left/Upper/Unit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [25.0, 15.0, 4.0].sliced(3, 1);

    // Setting cblas.Diag.Unit assumes diagonals are 1s, even if they are not
    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3).universal;
    auto b = [2.0, 3, 4].sliced(3, 1).universal;
    trmm(cblas.Side.Left, cblas.Uplo.Upper, cblas.Diag.Unit, 1.0, a, b);
    assert(b.equal(result));
}

/// Example: Right/Upper/Unit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [2.0, 13, 17].sliced(1, 3);

    // Setting cblas.Diag.Unit assumes diagonals are 1s, even if they are not
    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3).universal;
    auto b = [2.0, 3, 4].sliced(1, 3).universal;
    trmm(cblas.Side.Right, cblas.Uplo.Upper, cblas.Diag.Unit, 1.0, a, b);
    assert(b.equal(result));
}

@trusted pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    alias S0 = trmm!(double, Contiguous, Contiguous);
    alias S1 = trmm!(double, Contiguous, Universal);
    alias S2 = trmm!(double, Universal, Contiguous);
    alias S3 = trmm!(double, Universal, Universal);
}

///
@trusted pure nothrow @nogc
void trsm(T,
    SliceKind kindA,
    SliceKind kindB,
    )(
    cblas.Side side,
    cblas.Uplo uplo,
    cblas.Diag diag,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(T*, 2, kindB) b,
    )
in
{
    assert(a.length!1 == a.length!0);
    assert(a.length == (side == cblas.Side.Left ? b.length!0 : b.length!1));
}
do
{
    static if (kindB == Universal)
    {
        if (b._stride!1 != 1)
        {
            assert(b._stride!0 == 1, "Matrix B must have a stride equal to 1.");
            return .trsm(
                side == cblas.Side.Left ? cblas.Side.Right : cblas.Side.Left,
                uplo == cblas.Uplo.Upper ? cblas.Uplo.Lower : cblas.Uplo.Upper,
                diag,
                alpha,
                a.transposed,
                b.transposed.assumeCanonical,
            );
        }
    }

    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            // UPLO describes the stored matrix before applying TransA.
            uplo = uplo == cblas.Uplo.Upper ? cblas.Uplo.Lower : cblas.Uplo.Upper;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;

    cblas.trsm(
        cblas.Order.RowMajor,
        side,
        uplo,
        transA ? cblas.Transpose.Trans : cblas.Transpose.NoTrans,
        diag,

        cast(cblas.blasint) b.length!0,
        cast(cblas.blasint) b.length!1,

        alpha,

        a.iterator,
        cast(cblas.blasint) a._stride,

        b.iterator,
        cast(cblas.blasint) b.matrixStride,
    );
}

@safe pure nothrow @nogc
unittest
{
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    alias S0 = trsm!(double, Contiguous, Contiguous);
    alias S1 = trsm!(double, Contiguous, Universal);
    alias S2 = trsm!(double, Universal, Contiguous);
    alias S3 = trsm!(double, Universal, Universal);
}

///
@trusted pure nothrow @nogc
void symv(T,
    SliceKind kindA,
    SliceKind kindX,
    SliceKind kindY,
    )(
    Uplo uplo,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 1, kindX) x,
    T beta,
    Slice!(T*, 1, kindY) y,
    )
in
{
    assert(a.length!0 == a.length!1);
    assert(a.length!1 == x.length);
    assert(a.length!0 == y.length);
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;
    cblas.symv(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,
        uplo,
        
        cast(cblas.blasint) x.length,

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        vectorBase(x),
        cast(cblas.blasint) x._stride,

        beta,

        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = symv!(double, Contiguous, Contiguous, Contiguous);
    alias D0 = symv!(double, Contiguous, Contiguous, Universal);
    alias S1 = symv!(double, Contiguous, Universal, Contiguous);
    alias D1 = symv!(double, Contiguous, Universal, Universal);
    alias S2 = symv!(double, Universal, Contiguous, Contiguous);
    alias D2 = symv!(double, Universal, Contiguous, Universal);
    alias S3 = symv!(double, Universal, Universal, Contiguous);
    alias D3 = symv!(double, Universal, Universal, Universal);
}

///
@trusted pure nothrow @nogc
void symm(T,
    SliceKind kindA,
    SliceKind kindB,
    SliceKind kindC,
    )(
    Side side,
    Uplo uplo,
    T alpha,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(const(T)*, 2, kindB) b,
    T beta,
    Slice!(T*, 2, kindC) c,
    )
in
{
    assert(a.length!1 == a.length!0);
    if (side == Side.Left)
    {
        assert(a.length!1 == b.length!0);
        assert(a.length!0 == c.length!0);
        assert(c.length!1 == b.length!1);
    }
    else
    {
        assert(a.length!1 == b.length!1);
        assert(a.length!0 == c.length!1);
        assert(c.length!0 == b.length!0);
    }
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            uplo = uplo == Uplo.Lower ? Uplo.Upper : Uplo.Lower;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }

    static if (kindB == Universal && kindC == Universal)
    {
        if (b._stride!1 != 1
         || c._stride!1 != 1)
        {
            b = b.transposed;
            c = c.transposed;
            side = side == Side.Left ? Side.Right : Side.Left;
        }

    }
    assert(b._stride!1 == 1 && c._stride!1 == 1, "Matrices B and C must be either both row-major or both column-major.");

    cblas.symm(
        cblas.Order.RowMajor,
        side,
        uplo,

        cast(cblas.blasint) c.length!0,
        cast(cblas.blasint) c.length!1, 

        alpha,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,

        b.iterator,
        cast(cblas.blasint) b.matrixStride,

        beta,

        c.iterator,
        cast(cblas.blasint) c.matrixStride,
    );
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = symm!(double, Contiguous, Contiguous, Contiguous);
    alias D0 = symm!(double, Contiguous, Contiguous, Universal);
    alias S1 = symm!(double, Contiguous, Universal, Contiguous);
    alias D1 = symm!(double, Contiguous, Universal, Universal);
    alias S2 = symm!(double, Universal, Contiguous, Contiguous);
    alias D2 = symm!(double, Universal, Contiguous, Universal);
    alias S3 = symm!(double, Universal, Universal, Contiguous);
    alias D3 = symm!(double, Universal, Universal, Universal);
}

/++
Return the index of the maximum absolute value in logical view order. For
inputs without NaNs, ties select the first logical maximum. Complex elements
use `abs(real) + abs(imaginary)`. An empty view returns zero, following the
CBLAS convention.

Results for NaN-containing inputs are unspecified and may depend on the BLAS
backend and view layout.
+/
@trusted pure nothrow @nogc
auto iamax(T,
    SliceKind kindX,
    )(
    Slice!(const(T)*, 1, kindX) x,
    )
{
    if (x._stride < 0 && x.length)
    {
        // A forward native call would choose the last logical maximum on ties.
        import mir.math.common: fabs;
        auto magnitude(T value)
        {
            static if (isFloatingPoint!T)
                return fabs(value);
            else
                return fabs(value.re) + fabs(value.im);
        }
        cblas.CBLAS_INDEX result = 0;
        auto maximum = magnitude(x[0]);
        foreach (i; 1 .. x.length)
        {
            auto value = magnitude(x[i]);
            if (value > maximum)
            {
                maximum = value;
                result = cast(cblas.CBLAS_INDEX) i;
            }
        }
        return result;
    }

    return cblas.iamax(
        cast(cblas.blasint) x.length,
        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = iamax!(double, Contiguous);
    alias D0 = iamax!(double, Universal);
}

///
@trusted pure @nogc nothrow
void syr(T,
    SliceKind kindA,
    SliceKind kindC,
    )(
    Uplo uplo,
    T alpha,
    Slice!(const(T)*, 1, kindA) a,
    Slice!(T*, 2, kindC) c,
    )
in
{
    assert(a.length!0 == c.length!0);
    assert(c.length!1 == c.length!0);
}
do
{
    static if (kindC == Universal)
    {
        if (c._stride!1 != 1)
        {
            c = c.transposed;
            uplo = uplo == Uplo.Lower ? Uplo.Upper : Uplo.Lower;
        }
        assert(c._stride!1 == 1, "Matrix C must have a stride equal to 1.");
    }
    else
        enum transC = false;
    cblas.syr(
        cblas.Order.RowMajor,
        uplo,

        cast(cblas.blasint) c.length!0,

        alpha,

        vectorBase(a),
        cast(cblas.blasint) a._stride,

        c.iterator,
        cast(cblas.blasint) c._stride,
    );
}

///
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto result = [
        [1.0, 2, 3],
        [0.0, 4, 6],
        [0.0, 0, 9]
    ].fuse;

    auto x = [1.0, 2, 3].sliced;
    auto output = slice!double([3, 3], 0);

    syr(Uplo.Upper, 1.0, x, output);
    assert(output.equal(result));
}

///
@safe pure
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: slice;
    import mir.ndslice.fuse: fuse;
    import mir.ndslice.slice: sliced;

    auto result = [
        [1.0, 0, 0],
        [2.0, 4, 0],
        [3.0, 6, 9]
    ].fuse;

    auto x = [1.0, 2, 3].sliced;
    auto output = slice!double([3, 3], 0);

    syr(Uplo.Lower, 1.0, x, output);
    assert(output.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = syr!(double, Contiguous, Contiguous);
    alias D0 = syr!(double, Contiguous, Universal);
    alias S2 = syr!(double, Universal, Contiguous);
    alias D2 = syr!(double, Universal, Universal);
}

///
@trusted pure @nogc nothrow
void spr(T,
    SliceKind kindA,
    string type
    )(
    T alpha,
    Slice!(const(T)*, 1, kindA) a,
    Slice!(StairsIterator!(T*, type)) c,
    )
  if (type == "+" || type == "-")
in
{
    assert(a.length == c.length);
}
do
{
    enum Uplo uplo = type == "-" ? Uplo.Upper : Uplo.Lower;
    cblas.spr(
        cblas.Order.RowMajor,
        uplo,

        cast(cblas.blasint) c.length!0,

        alpha,

        vectorBase(a),
        cast(cblas.blasint) a._stride,

        c.iterator._iterator
    );
}

///
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [1.0, 2, 4, 3, 6, 9].stairs!"+"(3);

    auto x = [1.0, 2, 3].sliced;
    auto output = uninitSlice!double(6).stairs!"+"(3);

    spr(1.0, x, output);
    assert(output.equal(result));
}

///
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [1.0, 2, 3, 4, 6, 9].stairs!"-"(3);

    auto x = [1.0, 2, 3].sliced;
    auto output = uninitSlice!double(6).stairs!"-"(3);

    spr(1.0, x, output);
    assert(output.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = spr!(double, Contiguous, "+");
    alias D0 = spr!(double, Contiguous, "-");
    alias S1 = spr!(double, Universal,"+");
    alias D1 = spr!(double, Universal, "-");
}

///
@trusted pure @nogc nothrow
void spmv(T,
    SliceKind kindX,
    SliceKind kindY,
    string type
    )(
    T alpha,
    Slice!(StairsIterator!(T*, type)) a,
    Slice!(const(T)*, 1, kindX) x,
    T beta,
    Slice!(T*, 1, kindY) y,
    )
  if (type == "+" || type == "-")
in
{
    assert(a.length == x.length);
    assert(x.length == y.length);
}
do
{
    enum Uplo uplo = type == "-" ? Uplo.Upper : Uplo.Lower;
    cblas.spmv(
        cblas.Order.RowMajor,
        uplo,

        cast(cblas.blasint) x.length,

        alpha,

        a.iterator._iterator,

        vectorBase(x),
        cast(cblas.blasint) x._stride,

        beta,

        vectorBase(y),
        cast(cblas.blasint) y._stride,
    );
}

///
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [8.0, 14, 8].sliced;

    auto A = [1.0, 2, 3, 1, 2, 1].stairs!"+"(3);
    auto x = [1.0, 2, 3].sliced;
    auto output = uninitSlice!double(3);

    spmv(1.0, A, x, 0.0, output);
    assert(output.equal(result));
}

///
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.allocation: uninitSlice;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [14.0, 10, 10].sliced;

    auto A = [1.0, 2, 3, 1, 2, 1].stairs!"-"(3);
    auto x = [1.0, 2, 3].sliced;
    auto output = uninitSlice!double(3);

    spmv(1.0, A, x, 0.0, output);
    assert(output.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = spmv!(double, Contiguous, Contiguous, "+");
    alias D0 = spmv!(double, Contiguous, Universal, "+");
    alias S1 = spmv!(double, Contiguous, Contiguous, "-");
    alias D1 = spmv!(double, Contiguous, Universal, "-");
    alias S2 = spmv!(double, Universal, Contiguous, "+");
    alias D2 = spmv!(double, Universal, Universal, "+");
    alias S3 = spmv!(double, Universal, Contiguous, "-");
    alias D3 = spmv!(double, Universal, Universal, "-");
}

///
@trusted pure nothrow @nogc
void trmv(T,
    SliceKind kindA,
    SliceKind kindX,
    )(
    cblas.Uplo uplo,
    cblas.Diag diag,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(T*, 1, kindX) x,
    )
in
{
    assert(a.length!1 == a.length!0);
    assert(a.length!0 == x.length!0);
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;

    cblas.trmv(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,
        uplo,
        cblas.Transpose.NoTrans,
        diag,

        cast(cblas.blasint) a.length,

        a.iterator,
        cast(cblas.blasint) a._stride,

        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

/// Example: Upper/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;

    auto result = [29.0, 18, 4].sliced(3);

    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3);
    auto b = [2.0, 3, 4].sliced(3);
    trmv(cblas.Uplo.Upper, cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Lower/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;

    auto result = [6.0, 16, 17].sliced(3);

    auto a = [3.0, 0, 0, 5, 2, 0, 2, 3, 1].sliced(3, 3);
    auto b = [2.0, 3, 4].sliced(3);
    trmv(cblas.Uplo.Lower, cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Upper/NonUnit, but transposing first (so only diagonal is used)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.slice: sliced;

    auto result = [6.0, 6, 4].sliced(3);

    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3).transposed;
    auto b = [2.0, 3, 4].sliced(3);
    trmv(cblas.Uplo.Upper, cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Upper/Unit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [25.0, 15, 4].sliced(3);

    // Setting cblas.Diag.Unit assumes diagonals are 1s, even if they are not
    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3);
    auto b = [2.0, 3, 4].sliced(3);
    trmv(cblas.Uplo.Upper, cblas.Diag.Unit, a, b);
    assert(b.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = trmv!(double, Contiguous, Contiguous);
    alias D0 = trmv!(double, Contiguous, Universal);
    alias S1 = trmv!(double, Universal, Contiguous);
    alias D1 = trmv!(double, Universal, Universal);
}

///
@trusted pure nothrow @nogc
void tpmv(T,
    SliceKind kindX,
    string type
    )(
    cblas.Diag diag,
    Slice!(StairsIterator!(T*, type)) a,
    Slice!(T*, 1, kindX) x,
    )
  if (type == "+" || type == "-")
in
{
    assert(a.length == x.length);
}
do
{
    enum Uplo uplo = type == "-" ? Uplo.Upper : Uplo.Lower;
    cblas.tpmv(
        cblas.Order.RowMajor,
        uplo,
        cblas.Transpose.NoTrans,
        diag,

        cast(cblas.blasint) a.length,

        a.iterator._iterator,

        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

/// Example: Upper/NonUnit
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [29.0, 18, 4].sliced(3);

    auto a = [3.0, 5, 2, 2, 3, 1].stairs!"-"(3);
    auto b = [2.0, 3, 4].sliced(3);
    tpmv(cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Lower/NonUnit
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [6.0, 16, 17].sliced(3);

    auto a = [3.0, 5, 2, 2, 3, 1].stairs!"+"(3);
    auto b = [2.0, 3, 4].sliced(3);
    tpmv(cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Upper/Unit
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [25.0, 15, 4].sliced(3);

    // Setting cblas.Diag.Unit assumes diagonals are 1s, even if they are not
    auto a = [3.0, 5, 2, 2, 3, 1].stairs!"-"(3);
    auto b = [2.0, 3, 4].sliced(3);
    tpmv(cblas.Diag.Unit, a, b);
    assert(b.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = tpmv!(double, Contiguous, "+");
    alias D0 = tpmv!(double, Contiguous, "-");
    alias S1 = tpmv!(double, Universal, "+");
    alias D1 = tpmv!(double, Universal, "-");
}

///
@trusted pure nothrow @nogc
void trsv(T,
    SliceKind kindA,
    SliceKind kindX,
    )(
    cblas.Uplo uplo,
    cblas.Diag diag,
    Slice!(const(T)*, 2, kindA) a,
    Slice!(T*, 1, kindX) x,
    )
in
{
    assert(a.length!1 == a.length!0);
    assert(a.length!0 == x.length!0);
}
do
{
    static if (kindA == Universal)
    {
        bool transA;
        if (a._stride!1 != 1)
        {
            a = a.transposed;
            transA = true;
        }
        assert(a._stride!1 == 1, "Matrix A must have a stride equal to 1.");
    }
    else
        enum transA = false;

    cblas.trsv(
        transA ? cblas.Order.ColMajor : cblas.Order.RowMajor,
        uplo,
        cblas.Transpose.NoTrans,
        diag,

        cast(cblas.blasint) a.length,

        a.iterator,
        cast(cblas.blasint) a._stride,

        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

/// Example: Upper/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;

    auto result = [2.0, 3, 4].sliced(3);

    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3);
    auto b = [29.0, 18, 4].sliced(3);
    trsv(cblas.Uplo.Upper, cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Lower/NonUnit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;

    auto result = [2.0, 3, 4].sliced(3);

    auto a = [3.0, 0, 0, 5, 2, 0, 2, 3, 1].sliced(3, 3);
    auto b = [6.0, 16, 17].sliced(3);
    trsv(cblas.Uplo.Lower, cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Upper/NonUnit, but transposing first (so only diagonal is used)
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.dynamic: transposed;
    import mir.ndslice.slice: sliced;

    auto result = [2.0, 3, 4].sliced(3);

    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3).transposed;
    auto b = [6.0, 6, 4].sliced(3);
    trsv(cblas.Uplo.Upper, cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Upper/Unit
@safe pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: universal;

    auto result = [2.0, 3, 4].sliced(3);

    // Setting cblas.Diag.Unit assumes diagonals are 1s, even if they are not
    auto a = [3.0, 5, 2, 0, 2, 3, 0, 0, 1].sliced(3, 3);
    auto b = [25.0, 15, 4].sliced(3);
    trsv(cblas.Uplo.Upper, cblas.Diag.Unit, a, b);
    assert(b.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = trsv!(double, Contiguous, Contiguous);
    alias D0 = trsv!(double, Contiguous, Universal);
    alias S1 = trsv!(double, Universal, Contiguous);
    alias D1 = trsv!(double, Universal, Universal);
}

///
@trusted pure nothrow @nogc
void tpsv(T,
    SliceKind kindX,
    string type
    )(
    cblas.Diag diag,
    Slice!(StairsIterator!(T*, type)) a,
    Slice!(T*, 1, kindX) x,
    )
  if (type == "+" || type == "-")
in
{
    assert(a.length == x.length);
}
do
{
    enum Uplo uplo = type == "-" ? Uplo.Upper : Uplo.Lower;
    cblas.tpsv(
        cblas.Order.RowMajor,
        uplo,
        cblas.Transpose.NoTrans,
        diag,

        cast(cblas.blasint) a.length,

        a.iterator._iterator,

        vectorBase(x),
        cast(cblas.blasint) x._stride,
    );
}

/// Example: Upper/NonUnit
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [2.0, 3, 4].sliced(3);

    auto a = [3.0, 5, 2, 2, 3, 1].stairs!"-"(3);
    auto b = [29.0, 18, 4].sliced(3);
    tpsv(cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Lower/NonUnit
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [2.0, 3, 4].sliced(3);

    auto a = [3.0, 5, 2, 2, 3, 1].stairs!"+"(3);
    auto b = [6.0, 16, 17].sliced(3);
    tpsv(cblas.Diag.NonUnit, a, b);
    assert(b.equal(result));
}

/// Example: Upper/Unit
pure nothrow
unittest
{
    import mir.algorithm.iteration: equal;
    import mir.ndslice.slice: sliced;
    import mir.ndslice.topology: stairs;

    auto result = [2.0, 3, 4].sliced(3);

    // Setting cblas.Diag.Unit assumes diagonals are 1s, even if they are not
    auto a = [3.0, 5, 2, 2, 3, 1].stairs!"-"(3);
    auto b = [25.0, 15, 4].sliced(3);
    tpsv(cblas.Diag.Unit, a, b);
    assert(b.equal(result));
}

@safe pure nothrow @nogc
unittest
{
    alias S0 = tpsv!(double, Contiguous, "+");
    alias D0 = tpsv!(double, Contiguous, "-");
    alias S1 = tpsv!(double, Universal, "+");
    alias D1 = tpsv!(double, Universal, "-");
}
// Reversed views retain logical pairing and leave skipped elements untouched.
@safe pure nothrow @nogc
unittest
{
    import std.math: fabs, sqrt;
    foreach (reverseX; [false, true])
    foreach (reverseY; [false, true])
    {
        double[8] sx = -99, sy = -99;
        auto x = sx[1 .. 6].sliced.universal.strided(2);
        auto y = sy[1 .. 6].sliced.universal.strided(2);
        if (reverseX) x = x.reversed;
        if (reverseY) y = y.reversed;
        x[0] = 1; x[1] = -2; x[2] = 3;
        y[0] = 4; y[1] = 5; y[2] = -6;
        assert(dot(x.lightConst, y.lightConst) == -24);
        assert(fabs(nrm2(x.lightConst) - sqrt(14.0)) < 1e-12);
        assert(asum(x.lightConst) == 6);
        assert(iamax(x.lightConst) == 2);
        axpy(2.0, x.lightConst, y);
        assert(y[0] == 6 && y[1] == 1 && y[2] == 0);
        copy(x.lightConst, y);
        scal(2.0, x);
        swap(x, y);
        assert(x[0] == 1 && x[1] == -2 && x[2] == 3);
        assert(y[0] == 2 && y[1] == -4 && y[2] == 6);
        x[0] = -9; x[1] = 9; x[2] = 3;
        assert(iamax(x.lightConst) == 0);
        auto empty = x[0 .. 0];
        assert(dot(empty.lightConst, empty.lightConst) == 0);
        assert(nrm2(empty.lightConst) == 0);
        assert(asum(empty.lightConst) == 0);
        assert(iamax(empty.lightConst) == 0);
        scal(5.0, empty);
        auto single = x[0 .. 1];
        assert(asum(single.lightConst) == 9);
        assert(iamax(single.lightConst) == 0);
        foreach (i; 0 .. 8)
            if (i != 1 && i != 3 && i != 5)
                assert(sx[i] == -99 && sy[i] == -99);
    }
}

// Complex absolute sums and maximum indices use |real| + |imaginary|.
@safe pure nothrow @nogc
unittest
{
    alias T = cblas._cdouble;
    T[5] storage = testValue!T(-99);
    auto x = storage[1 .. 4].sliced.universal.reversed;
    x[0] = testValue!T(3, 4); x[1] = testValue!T(-3, -4); x[2] = testValue!T(6, 0);
    assert(iamax(x.lightConst) == 0); // magnitude squared would select index 2
    assert(asum(x.lightConst) == 20);
    import std.math: fabs, sqrt;
    assert(fabs(nrm2(x.lightConst) - sqrt(86.0)) < 1e-12);
    T[3] output;
    auto y = output[].sliced;
    copy(x.lightConst, y);
    scal(testValue!T(2), x);
    axpy(testValue!T(-1), y.lightConst, x);
    foreach (i; 0 .. 3) assert(x[i] == y[i]);
    assert(storage[0] == testValue!T(-99) && storage[4] == testValue!T(-99));
}

// Exercise every level-2 vector boundary against its positive-stride result.
// Constructing packed StairsIterator fixtures requires system access.
@system pure nothrow @nogc
unittest
{
    import std.math: fabs;
    foreach (reverseX; [false, true])
    foreach (reverseY; [false, true])
    {
        double[8] sx = -99, sy = -99;
        double[3] rx = [1, 2, 3], ry = [4, 5, 6];
        auto x = sx[1 .. 6].sliced.universal.strided(2);
        auto y = sy[1 .. 6].sliced.universal.strided(2);
        if (reverseX) x = x.reversed;
        if (reverseY) y = y.reversed;
        foreach (i; 0 .. 3) { x[i] = rx[i]; y[i] = ry[i]; }
        auto xr = rx[].sliced;
        auto yr = ry[].sliced;
        double[9] sa = [2, 1, 3, 1, 4, 2, 3, 2, 5], ra = sa;
        auto a = sa[].sliced(3, 3);
        auto ar = ra[].sliced(3, 3);
        ger(2.0, x.lightConst, y.lightConst, a);
        ger(2.0, xr.lightConst, yr.lightConst, ar);
        assert(sa == ra);
        gerc(1.0, x.lightConst, y.lightConst, a);
        gerc(1.0, xr.lightConst, yr.lightConst, ar);
        assert(sa == ra);
        syr(Uplo.Upper, 2.0, x.lightConst, a);
        syr(Uplo.Upper, 2.0, xr.lightConst, ar);
        assert(sa == ra);
        gemv(2.0, a.lightConst, x.lightConst, 3.0, y);
        gemv(2.0, ar.lightConst, xr.lightConst, 3.0, yr);
        foreach (i; 0 .. 3) assert(y[i] == yr[i]);
        symv(Uplo.Upper, 2.0, a.lightConst, x.lightConst, 3.0, y);
        symv(Uplo.Upper, 2.0, ar.lightConst, xr.lightConst, 3.0, yr);
        foreach (i; 0 .. 3) assert(y[i] == yr[i]);
        trmv(Uplo.Upper, cblas.Diag.NonUnit, a.lightConst, x);
        trmv(Uplo.Upper, cblas.Diag.NonUnit, ar.lightConst, xr);
        foreach (i; 0 .. 3) assert(x[i] == xr[i]);
        trsv(Uplo.Upper, cblas.Diag.NonUnit, a.lightConst, x);
        trsv(Uplo.Upper, cblas.Diag.NonUnit, ar.lightConst, xr);
        foreach (i; 0 .. 3) assert(fabs(x[i] - xr[i]) < 1e-12);
        double[6] packed = [2, 3, 4, 5, 6, 7], packedRef = packed;
        auto p = packed[].sliced.stairs!"-"(3);
        auto pr = packedRef[].sliced.stairs!"-"(3);
        spr(2.0, x.lightConst, p);
        spr(2.0, xr.lightConst, pr);
        assert(packed == packedRef);
        spmv(2.0, p, x.lightConst, 3.0, y);
        spmv(2.0, pr, xr.lightConst, 3.0, yr);
        foreach (i; 0 .. 3) assert(fabs(y[i] - yr[i]) < 1e-8);
        tpmv(cblas.Diag.NonUnit, p, x);
        tpmv(cblas.Diag.NonUnit, pr, xr);
        foreach (i; 0 .. 3) assert(fabs(x[i] - xr[i]) < 1e-10);
        tpsv(cblas.Diag.NonUnit, p, x);
        tpsv(cblas.Diag.NonUnit, pr, xr);
        foreach (i; 0 .. 3) assert(fabs(x[i] - xr[i]) < 1e-12);
        foreach (i; 0 .. 8)
            if (i != 1 && i != 3 && i != 5)
                assert(sx[i] == -99 && sy[i] == -99);
    }
}
