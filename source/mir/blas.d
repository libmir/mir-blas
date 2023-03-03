/++
Low-level ndslice wrapper for BLAS.

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

        x.iterator,
        cast(cblas.blasint) x._stride,

        y.iterator,
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
    return cblas.nrm2(
        cast(cblas.blasint) x.length,
        x.iterator,
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
    return cblas.asum(
        cast(cblas.blasint) x.length,
        x.iterator,
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
        x.iterator,
        cast(cblas.blasint) x._stride,
        y.iterator,
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
    cblas.scal(
        cast(cblas.blasint) x.length,
        a,
        x.iterator,
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
        x.iterator,
        cast(cblas.blasint) x._stride,
        y.iterator,
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
        x.iterator,
        cast(cblas.blasint) x._stride,
        y.iterator,
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

        cast(cblas.blasint) a.length!0,
        cast(cblas.blasint) a.length!1,

        alpha,

        x.iterator,
        cast(cblas.blasint) x._stride,

        y.iterator,
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

        cast(cblas.blasint) a.length!0,
        cast(cblas.blasint) a.length!1,

        alpha,

        x.iterator,
        cast(cblas.blasint) x._stride,

        y.iterator,
        cast(cblas.blasint) y._stride,

        a.iterator,
        cast(cblas.blasint) a.matrixStride,
    );
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

        x.iterator,
        cast(cblas.blasint) x._stride,

        beta,

        y.iterator,
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

        x.iterator,
        cast(cblas.blasint) x._stride,

        beta,

        y.iterator,
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

///
@trusted pure nothrow @nogc
auto iamax(T,
    SliceKind kindX,
    )(
    Slice!(const(T)*, 1, kindX) x,
    )
{
    return cblas.iamax(
        cast(cblas.blasint) x.length,
        x.iterator,
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

        a.iterator,
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

        a.iterator,
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

        x.iterator,
        cast(cblas.blasint) x._stride,

        beta,

        y.iterator,
        cast(cblas.blasint) y._stride,
    );
}

///
@trusted pure
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
@trusted pure
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
