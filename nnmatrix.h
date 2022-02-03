#ifndef _nn_matrix_h_
#define _nn_matrix_h_

#include <string>
#include <cstddef>
#include <vector>
#include <type_traits>
#include <iostream>
#include <stdexcept>
#include <complex>

#include <utilfuncs/utilfuncs.h>

//---------------------------------------------------------------------------------------------------
template<typename T> concept bool IsNmberOrComplex()
	{
		return (std::is_convertible<T, double>::value || std::is_convertible<T, std::complex<decltype(T())>>::value);
	}
//---------------------------------------------------------------------------------------------------
template<IsNmberOrComplex T> struct NNMatrix
{
	struct NNVector : public std::vector<T>
	{
		virtual ~NNVector(){}
		const T& operator[](size_t p) const { return std::vector<T>::at(p); }
		T& operator[](size_t p) { return std::vector<T>::at(p); }
	};
	std::vector< NNVector > mat;
	size_t nrows, ncols;

	NNMatrix(size_t nRows=1, size_t nCols=1) : nrows(nRows), ncols(nCols) { InitZero(); }
	NNMatrix(const NNMatrix<T> &M) { (*this)=M; }
	virtual ~NNMatrix(){}
	
	NNMatrix<T>& operator=(const NNMatrix<T> &M)
	{
		clear();
		nrows=M.nrows;
		ncols=M.ncols;
		for (auto v:M.mat) mat.push_back(v);
		return *this;
	}

	NNMatrix<T>& operator=(const NNVector &V)
	{
		SetRowCol(1, V.size());
		for (size_t i=0; i<V.size(); i++) mat[0][i]=V[i];
		return *this;
	}

	void clear() { for (auto v:mat) v.clear(); mat.clear(); nrows=ncols=0; }
	
	void SetRowCol(size_t r, size_t c) { clear(); if ((r>0)&&(c>0)) { nrows=r; ncols=c; InitZero(); }}
	void SetSquare(size_t rc) { clear(); if (rc>0) { nrows=ncols=rc; InitZero(); }}
	
	T& at(size_t r, size_t c) { if ((r<nrows)&&(c<ncols)) return mat[r][c]; else throw std::out_of_range("NNMatrix::at()"); }
	NNVector& operator[](size_t r) { if (r<nrows) return mat.at(r); else throw std::out_of_range("NNMatrix::operator[]()"); }
	const NNVector& operator[](size_t r) const { if (r<nrows) return mat.at(r); else throw std::out_of_range("NNMatrix::operator[]()"); }

	bool IsSquare() const { return ((nrows>0)&&(nrows==ncols)); }
	bool IsEmpty() const { return ((nrows==0)||(ncols==0)); }
	bool IsZero() const
	{
		bool b=false;
		if (!IsEmpty())
		{
			size_t r=0, c;
			do { c=0; do { b=(mat[r][c]==0); c++; } while (b&&(c<ncols)); r++; } while (b&&(r<nrows));
		}
		return b;
	}
	bool IsIdentity() const
	{
		bool b=false;
		if (IsSquare())
		{
			size_t r=0, c;
			do { c=0; do { b=(c==r)?(mat[r][c]==1):(mat[r][c]==0); c++; } while (b&&(c<ncols)); r++; } while (b&&(r<nrows));
		}
		return b;
	}
	bool IsSameSize(const NNMatrix<T> &M) const { return ((nrows==M.nrows)&&(ncols==M.ncols)); } //types: this T <= M T???
	
	void InitEmpty() { clear(); }
	void InitZero()
	{
		for (auto v:mat) v.clear(); mat.clear();
		for (size_t r=0;r<nrows;r++)
		{
			NNVector V;
			for (size_t c=0;c<ncols;c++) V.push_back(0);
			mat.push_back(V);
		}
	}
	bool InitIdentity()
	{
		for (auto v:mat) v.clear(); mat.clear();
		if (IsSquare())
		{
			for (size_t r=0;r<nrows;r++)
			{
				NNVector V;
				for (size_t c=0;c<ncols;c++) V.push_back((r==c)?1:0);
				mat.push_back(V);
			}
		}
		else return false;
		return true;
	}
	
	NNMatrix<T> Identity() const { NNMatrix<T> R(*this); R.InitIdentity(); return R; }

	NNMatrix<T> Diagonal() const // |\| TL to BR - as a single-row matrix
	{
		NNMatrix<T> R;
		if (IsSquare())
		{
			R.SetRowCol(nrows, ncols);
			R.InitZero();
			for (size_t i=0; i<ncols; i++) R[i][i]=mat[i][i];
		}
		return R;
	}

	NNMatrix<T> AntiDiagonal() const
	{
		NNMatrix<T> R;
		if (IsSquare()) // { R.SetRowCol(1, nrows); for (size_t r=0;r<nrows;r++) R[0][r]=mat[r][nrows-r-1]; }
		{
			R.SetRowCol(nrows, ncols);
			R.InitZero();
			for (size_t i=0; i<ncols; i++) R[i][ncols-i-1]=mat[i][ncols-i-1];
		}
		return R;
	}

	double Trace() const { double d=0.0; auto R=Diagonal(); for (auto n: R[0]) d+=n; return d; }
	
	NNMatrix<T>& Transpose()
	{
		NNMatrix<T> R(ncols, nrows);
		for (size_t r=0;r<nrows;r++) for (size_t c=0;c<ncols;c++) R.mat[c][r]=mat[r][c];
		*this=R;
		return *this;
	}

	void SwapRows(size_t rownum, size_t pos)
	{
		if ((rownum<nrows)&&(pos<nrows))
		{
			NNVector va=mat[rownum];	NNVector vb=mat[pos];
			mat.erase(mat.begin()+pos);		mat.insert(mat.begin()+pos, va);
			mat.erase(mat.begin()+rownum);	mat.insert(mat.begin()+rownum, vb);
		}
	}

	void SwapColumns(size_t colnum, size_t pos) { if ((colnum<ncols)&&(pos<ncols)) { Transpose(); SwapRows(colnum, pos); Transpose(); } }

	void RemoveRow(size_t rownum) { if (rownum<nrows) { mat.erase(mat.begin()+rownum); nrows--; } }
	void RemoveColumn(size_t colnum) { Transpose(); RemoveRow(colnum); Transpose(); }

	void MulRowWith(size_t rownum, T m)		{ if (rownum<nrows) { for (auto& n:mat[rownum]) n*=m; } }
	void AddRows(size_t rto, size_t radd)	{ if ((rto<nrows)&&(radd<nrows)) { for (size_t c=0;c<ncols;c++) mat[rto][c]+=mat[radd][c]; } }
	void SubRows(size_t rfrom, size_t rsub)	{ if ((rfrom<nrows)&&(rsub<nrows)) { for (size_t c=0;c<ncols;c++) mat[rfrom][c]-=mat[rsub][c]; } }

	NNMatrix<T>& Translate(T t) //adds t to every [i][j]
	{
		for (size_t r=0;r<nrows;r++) for (size_t c=0;c<ncols;c++) mat[r][c]+=t;
		return *this;
	}
	
	virtual std::string AsPrintable() const //returns a printable string
	{
		std::string s="";
		bool bInt = std::is_integral<T>::value;
		for (auto v:mat)
		{
			for (auto n:v) if (bInt) sayss(s, ttos<T>(n).c_str(), "\t"); else sayss(s, n, "\t");
			sayss(s, "\n");
		}
		return s;
	}

	NNMatrix<T>& Concatenate(const NNMatrix<T> &M)
	{
		if (!M.IsEmpty()&&(IsEmpty()||(nrows==M.nrows)))
		{
			NNMatrix<T> R;
			size_t r, c, nr=((nrows)?nrows:M.nrows), nc=(ncols+M.ncols);
			
			R.SetRowCol(nr, nc);
			for (r=0;r<nrows;r++) for (c=0;c<ncols;c++) R[r][c]=mat[r][c];
			for (r=0;r<M.nrows;r++) for (c=0;c<M.ncols;c++) R[r][c+ncols]=M[r][c];
			*this=R;
		}
		return *this;
	}

	NNMatrix<T>& operator*=(T n) { for (auto& v:mat) for (auto& t:v) t*=n; return *this; }
	NNMatrix<T>& operator*=(const NNMatrix<T> &M)
	{
		if (ncols==M.nrows)
		{
			NNMatrix<T> R(nrows, M.ncols);
			NNVector cv;
			T n;
			auto mulvecs=[&](const NNVector &v1, const NNVector &v2)->T { T n=0; for (size_t i=0;i<v1.size();i++) n+=(v1[i]*v2[i]); return n; };
			
			for (size_t r=0;r<nrows;r++)
			{
				for (size_t c=0;c<M.ncols;c++)
				{
					cv.clear();
					for (size_t mr=0;mr<M.nrows;mr++) cv.push_back(M.mat[mr][c]);
					R.mat[r][c]=mulvecs(mat[r],cv);
				}
			}
			*this=R;
		}
		return *this;
	}
	
	NNMatrix<T>& HadamardProduct(const NNMatrix<T> &M)
	{
		if (!IsEmpty()&&!M.IsEmpty()&&IsSameSize(M))
		{
			for (size_t r=0;r<nrows;r++) for (size_t c=0;c<ncols;c++) mat[r][c]*=M[r][c];
		}
		return *this;
	}
	
	NNMatrix<T>& KroneckerProduct(const NNMatrix<T> &M) //tensor product
	{
		NNMatrix<T> R, W;
		auto append_mat=[&](NNMatrix<T> &R, NNMatrix<T> &W, size_t rpos, size_t cpos)
						{ for (size_t r=0;r<W.nrows;r++) for (size_t c=0;c<W.ncols;c++) { R[r+rpos][c+cpos]=W[r][c]; }};
		if (!IsEmpty()&&!M.IsEmpty())
		{
			R.SetRowCol((nrows*M.nrows), (ncols*M.ncols));
			size_t cc;
			for (size_t r=0;r<nrows;r++)
			{
				cc=0;
				for (size_t c=0;c<ncols;c++) { W=M; W*=mat[r][c]; append_mat(R, W, r*M.nrows, cc); cc+=M.ncols; }
			}
		}
		*this=R;
		return *this;
	}

	NNMatrix<T>& KroneckerSum(const NNMatrix<T> &M) //tensor sum
	{
		NNMatrix<T> A, B;
		if (!IsEmpty()&&!M.IsEmpty())
		{
			A=*this; A.KroneckerProduct(M.Identity());
			B=Identity(); B.KroneckerProduct(M);
			*this=(A+B);
		}
		return *this;
	}
	
	NNMatrix<T>& operator+=(const NNMatrix<T> &M)
	{
		if (IsSameSize(M))
		{
			for (size_t r=0;r<nrows;r++) for (size_t c=0;c<ncols;c++) mat[r][c]+=M.mat[r][c];
		}
		return *this;
	}

	NNMatrix<T>& operator-=(const NNMatrix<T> &M) { auto R=M; R*=-1; *this+=R; return *this; }

	const NNMatrix<T> Minor(size_t nr, size_t nc)
	{
		NNMatrix<T> R(*this);
		if (IsSquare()&&(nrows>2)&&(nr<nrows)&&(nc<ncols)) { R.RemoveRow(nr); R.RemoveColumn(nc); }
		else throw std::out_of_range("NNMatrix::Minor()");
		return R;
	}

	T Determinant() //brute force
	{
		T d=0.0;
		if (IsSquare()) //not 0x0
		{
			auto is_big=[](const NNMatrix<T> &M) -> bool { return (M.IsSquare()&&(M.nrows>2)); };
			auto not_zero=[](T t)->bool{ return !((t*t)==T{}); };
			if (nrows==2) { d=((mat[0][0]*mat[1][1])-(mat[0][1]*mat[1][0])); } //2x2 - done
			else if (is_big(*this)) //nxn - expand
			{
				for (size_t c=0;c<ncols;c++)
				{
					T t=mat[0][c];
					if (not_zero(t))
					{
						bool bplus=((c%2)==0);
						auto M=Minor(0,c);
						T dm=M.Determinant();
						dm*=t;
						if (bplus) d+=dm; else d-=dm;
					}
				}
			}
			else d=mat[0][0]; //1x1
		}
		return d;
	}

};

//---------------------------------------------------------------------------------------------------
template<typename T> void PrintMatrix(const NNMatrix<T> &M) { std::cout << "\n" << M.AsPrintable() << "\n"; } //CLI-only (use AsPrintable() elsewhere)

//---------------------------------------------------------------------------------------------------
template<typename T> bool operator==(const NNMatrix<T> &L, const NNMatrix<T> &R)
{
	bool b=false;
	if ((L.nrows==R.nrows)&&(L.ncols==R.ncols))
	{
		size_t r=0, c;
		do { c=0; do { b=(L[r][c]==R[r][c]); c++; } while (b&&(c<L.ncols)); r++; } while (b&&(r<L.nrows));
	}
	return b;
}

template<typename T> bool operator!=(const NNMatrix<T> &L, const NNMatrix<T> &R) { return !(L==R); }
template<typename T> NNMatrix<T> operator+(const NNMatrix<T> &L, const NNMatrix<T> &R) { auto M=L; M+=R; return M; }
template<typename T> NNMatrix<T> operator-(const NNMatrix<T> &L, const NNMatrix<T> &R) { auto M=L; M-=R; return M; }
template<typename T> NNMatrix<T> operator*(const NNMatrix<T> &L, const NNMatrix<T> &R) { auto M=L; M*=R; return M; }

template<typename T> NNMatrix<T> RowVector(const NNMatrix<T> &M, size_t row)
{
	auto R=M;
	R.SetRowCol(1, M.ncols);
	for (auto v:R.mat) v.clear(); R.mat.clear();
	if (row<M.nrows) R.mat.push_back(M.mat[row]);
	return R;
}

template<typename T> NNMatrix<T> ColumnVector(const NNMatrix<T> &M, size_t col)
{
	auto R=M;
	R.SetRowCol(M.nrows, 1);
	if (col<M.ncols) for (size_t i=0;i<M.nrows;i++) R.mat[i][0]=M.mat[i][col];
	return R;
}



#endif
