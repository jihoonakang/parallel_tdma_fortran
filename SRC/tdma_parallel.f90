!>
!> @brief       Parallel tri-diagonal matrix solver using cyclic reduction (CR), parallel CR (PCR),
!>              and Thomas+PCR hybrid algorithm
!> @details     The CR algorithm is described on Parallel Scientific Computing in C++ and MPI
!>              by Karniadakis and Kirby. CR algorithm removes odd rows recursively,
!>              so MPI processes begin to drop out after single row is left per MPI process,
!>              while PCR can use full parallelism. Therefore, PCR is a good solution from
!>              the level where single row is left per MPI process. In this implementation,
!>              we can choose CR or PCR algorithm from the single-row level.
!>              Odd-rows are removed successively and we obtain two reduced equations finally.
!>              Obtained solutions from 2x2 matrix equations are used to obtain other unknowns.
!>              Hybrid Thomas-PCR algorithm is from the work of Laszlo, Gilles and Appleyard, 
!>              Manycore Algorithms for Batch Scalar and Block Tridiagonal Solvers, ACM TOMS, 
!>              42, 31 (2016).
!>
!> @author      Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
!> @date        20 January 2019
!> @version     0.1
!> @par         Copyright
!>              Copyright (c) 2018 by Ji-Hoon Kang. All rights reserved.
!> @par         License     
!>              This project is release under the terms of the MIT License (see LICENSE in )
!>

module tdma_parallel

    use mpi

    implicit none

    integer(4), private :: n_mpi                    !< Number of rows per MPI process and should be 2^n.
    integer(4), private :: nprocs                   !< Number of MPI process and should be also 2^m.
    integer(4), private :: myrank                   !< MPI process ID

    real(8), pointer, dimension(:), private :: a    !< Local private pointer for coefficient maxtix a
    real(8), pointer, dimension(:), private :: b    !< Local private pointer for coefficient maxtix a
    real(8), pointer, dimension(:), private :: c    !< Local private pointer for coefficient maxtix a
    real(8), pointer, dimension(:), private :: r    !< Local private pointer for RHS vector r
    real(8), pointer, dimension(:), private :: x    !< Local private pointer for solution vector x

    contains

    !>
    !> @brief   Initialize local private variables from global input parameters.
    !> @param   n Size of global array
    !> @param   np_world Number of MPI process
    !> @param   rank_world rank ID in MPI_COMM_WORLD
    !>
    subroutine tdma_setup(n, np_world, rank_world)

        integer(4), intent(in) :: n
        integer(4), intent(in) :: np_world
        integer(4), intent(in) :: rank_world

        nprocs = np_world
        myrank = rank_world
        n_mpi  = n / nprocs

    end subroutine tdma_setup

    !>
    !> @brief   CR solver: cr_forward_multiple + cr_forward_single + cr_backward_single + cr_backward_multiple
    !> @param   a_mpi (input) Lower off-diagonal coeff., which is assigned to local private pointer a
    !> @param   b_mpi (input) Diagonal coeff., which is assigned to local private pointer b
    !> @param   c_mpi (input) Upper off-diagonal coeff.,, which is assigned to local private pointer c
    !> @param   r_mpi (input) RHS vector, which is assigned to local private pointer r
    !> @param   x_mpi (output) Solution vector, which is assigned to local private pointer x
    !>
    subroutine cr_solver(a_mpi, b_mpi, c_mpi, r_mpi, x_mpi)

        implicit none

        real(8), intent(inout), target :: a_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: b_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: c_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: r_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: x_mpi(0:n_mpi+1)

        a => a_mpi
        b => b_mpi
        c => c_mpi
        r => r_mpi
        x => x_mpi

        call cr_forward_multiple_row
        call cr_forward_single_row ! Including 2x2 solver
        call cr_backward_single_row
        call cr_backward_multiple_row

        nullify(a, b, c, r, x)

    end subroutine cr_solver

    !>
    !> @brief   CR-PCR solver: cr_forward_multiple + pcr_forward_single + cr_backward_multiple
    !> @param   a_mpi (input) Lower off-diagonal coeff., which is assigned to local private pointer a
    !> @param   b_mpi (input) Diagonal coeff., which is assigned to local private pointer b
    !> @param   c_mpi (input) Upper off-diagonal coeff.,, which is assigned to local private pointer c
    !> @param   r_mpi (input) RHS vector, which is assigned to local private pointer r
    !> @param   x_mpi (output) Solution vector, which is assigned to local private pointer x
    !>
    subroutine cr_pcr_solver(a_mpi, b_mpi, c_mpi, r_mpi, x_mpi)

        implicit none

        real(8), intent(inout), target :: a_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: b_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: c_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: r_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: x_mpi(0:n_mpi+1)

        a => a_mpi
        b => b_mpi
        c => c_mpi
        r => r_mpi
        x => x_mpi

        call cr_forward_multiple_row 
        call pcr_forward_single_row       ! Including 2x2 solver
        call cr_backward_multiple_row

        nullify(a, b, c, r, x)

    end subroutine cr_pcr_solver

    !>
    !> @brief   Thomas-PCR solver: pThomas_forward_multiple + pcr_forward_double + pThomas_backward_multiple
    !> @param   a_mpi (input) Lower off-diagonal coeff., which is assigned to local private pointer a
    !> @param   b_mpi (input) Diagonal coeff., which is assigned to local private pointer b
    !> @param   c_mpi (input) Upper off-diagonal coeff.,, which is assigned to local private pointer c
    !> @param   r_mpi (input) RHS vector, which is assigned to local private pointer r
    !> @param   x_mpi (output) Solution vector, which is assigned to local private pointer x
    !>
    subroutine Thomas_pcr_solver(a_mpi, b_mpi, c_mpi, r_mpi, x_mpi)

        implicit none

        real(8), intent(inout), target :: a_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: b_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: c_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: r_mpi(0:n_mpi+1)
        real(8), intent(inout), target :: x_mpi(0:n_mpi+1)

        a => a_mpi
        b => b_mpi
        c => c_mpi
        r => r_mpi
        x => x_mpi

        call pThomas_forward_multiple_row()
        call pcr_double_row_substitution()

        nullify(a, b, c, r, x)

    end subroutine Thomas_pcr_solver

    !> 
    !> @brief   Forward elimination of CR until a single row per MPI process remains.
    !> @details After a single row per MPI process remains, PCR or CR between a single row is performed.
    !>   
    subroutine cr_forward_multiple_row

        use mpi

        implicit none

        integer(4)  :: i, l
        integer(4)  :: nlevel       
        integer(4)  :: ip, in, start, dist_row, dist2_row
        real(8)     :: alpha, gamma
        real(8)     :: sbuf(4), rbuf(4)

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(2)
        integer(4)  :: ierr

        nlevel      = log(dble(n_mpi))/log(dble(2))         ! Variable nlevel is used to indicates when single row remains.
        dist_row    = 1
        dist2_row   = 2

        do l = 0, nlevel-1
            start = dist2_row
            ! Data exchange is performed using MPI send/recv for each succesive reduction
            if(myrank < nprocs-1) then
                call mpi_irecv(rbuf, 4, MPI_REAL8, myrank+1, 0, MPI_COMM_WORLD, request(1), ierr)
            endif

            if(myrank > 0) then
                sbuf(1) = a(dist_row)
                sbuf(2) = b(dist_row)
                sbuf(3) = c(dist_row)
                sbuf(4) = r(dist_row)
                call mpi_isend(sbuf, 4, MPI_REAL8, myrank-1, 0, MPI_COMM_WORLD, request(2), ierr)
            endif

            if(myrank < nprocs-1) then
                call mpi_wait(request(1), status, ierr)
                a(n_mpi+1) = rbuf(1)
                b(n_mpi+1) = rbuf(2)
                c(n_mpi+1) = rbuf(3)
                r(n_mpi+1) = rbuf(4)
            endif

            ! Odd rows of remained rows are reduced to even rows of remained rows in each reduction step.
            ! Index in of global last row is out of range, but we treat it as a = c = r = 0 and b = 1 in main function.
            do i = start, n_mpi, dist2_row
                ip = i - dist_row
                in = min(i+dist_row, n_mpi+1)
                alpha = -a(i) / b(ip)
                gamma = -c(i) / b(in)

                b(i) = b(i) + alpha * c(ip) + gamma * a(in)
                a(i) = alpha * a(ip)
                c(i) = gamma * c(in)
                r(i) = r(i) + alpha * r(ip) + gamma * r(in)
            enddo

            ! As reduction continues, the indices of required coefficients doubles.
            dist2_row = dist2_row * 2
            dist_row  = dist_row * 2

            if(myrank > 0) then
                call mpi_wait(request(2), status, ierr)
            endif

        enddo

    end subroutine cr_forward_multiple_row

    !>
    !> @brief   Backward substitution of CR after single-row solution per MPI process is obtained.
    !>
    subroutine cr_backward_multiple_row

        use mpi

        implicit none

        integer(4)  :: i, l
        integer(4)  :: nlevel
        integer(4)  :: ip, in, dist_row, dist2_row

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(2)
        integer(4)  :: ierr

        nlevel      = log(dble(n_mpi))/log(dble(2))
        dist_row    = n_mpi / 2

        ! Each rank requires a solution on last row of previous rank.
        if(myrank > 0) then
            call mpi_irecv(x(0), 1, MPI_REAL8, myrank-1, 100, MPI_COMM_WORLD, request(1),ierr)
        endif

        if(myrank < nprocs-1) then
            call mpi_isend(x(n_mpi), 1, MPI_REAL8, myrank+1, 100, MPI_COMM_WORLD, request(2),ierr)
        endif

        if(myrank > 0) then
            call mpi_wait(request(1), status,ierr)
        endif

        do l = nlevel-1, 0, -1
            dist2_row = dist_row * 2
            do i = n_mpi-dist_row, 0, -dist2_row
                ip = i - dist_row
                in = i + dist_row
                x(i) = r(i)-c(i)*x(in) - a(i)*x(ip)
                x(i) = x(i) / b(i)
            enddo
            dist_row = dist_row / 2
        enddo

        if(myrank < nprocs-1) then
            call mpi_wait(request(2), status,ierr)
        endif

    end subroutine cr_backward_multiple_row

    !> 
    !> @brief   Forward elimination of CR between a single row per MPI process.
    !>
    subroutine cr_forward_single_row

        use mpi

        implicit none

        integer(4)  :: i, l
        integer(4)  :: nlevel, nhprocs
        integer(4)  :: ip, in, dist_rank, dist2_rank
        real(8)     :: alpha, gamma, det
        real(8)     :: sbuf(4), rbuf0(4), rbuf1(4)

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(4)
        integer(4)  :: ierr

        nlevel      = log(dble(nprocs))/log(dble(2))
        nhprocs     = nprocs / 2

        dist_rank  = 1
        dist2_rank = 2

        ! Cyclic reduction continues until 2x2 matrix are made in rank of nprocs-1 and nprocs/2-1
        do l = 0, nlevel-2

            ! Odd rows of remained rows are reduced to even rows of remained rows in each reduction step.
            ! Coefficients are updated for even rows only.

            print *, myrank , l

            if(mod((myrank+1),dist2_rank) == 0) then
                if(myrank-dist_rank>=0) then
                    call mpi_irecv(rbuf0, 4, MPI_REAL8, myrank-dist_rank, 400, MPI_COMM_WORLD, request(3), ierr)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_irecv(rbuf1, 4, MPI_REAL8, myrank+dist_rank, 401, MPI_COMM_WORLD, request(4), ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_wait(request(3), status, ierr)
                    a(0) = rbuf0(1)
                    b(0) = rbuf0(2)
                    c(0) = rbuf0(3)
                    r(0) = rbuf0(4)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_wait(request(4), status, ierr)
                    a(n_mpi+1) = rbuf1(1)
                    b(n_mpi+1) = rbuf1(2)
                    c(n_mpi+1) = rbuf1(3)
                    r(n_mpi+1) = rbuf1(4)
                endif

                i = n_mpi
                ip = 0
                in = i + 1
                alpha = -a(i) / b(ip)
                gamma = -c(i) / b(in)

                b(i) = b(i) + (alpha * c(ip) + gamma * a(in))
                a(i) = alpha * a(ip)
                c(i) = gamma * c(in)
                r(i) = r(i) + (alpha * r(ip) + gamma * r(in))

            else if(mod((myrank+1),dist2_rank) == dist_rank) then

                sbuf(1) = a(n_mpi)
                sbuf(2) = b(n_mpi)
                sbuf(3) = c(n_mpi)
                sbuf(4) = r(n_mpi)

                if(myrank+dist_rank<nprocs) then
                    call mpi_isend(sbuf, 4, MPI_REAL8, myrank+dist_rank, 400, MPI_COMM_WORLD, request(1), ierr);
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_isend(sbuf, 4, MPI_REAL8, myrank-dist_rank, 401, MPI_COMM_WORLD, request(2), ierr);
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_wait(request(1), status, ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_wait(request(2), status, ierr)
                endif
            endif

            dist_rank  = dist_rank * 2
            dist2_rank = dist2_rank * 2

        enddo

        ! Solving 2x2 matrix. Rank of nprocs-1 and nprocs/2-1 solves it simultaneously.

        if(myrank==nhprocs-1) then
            call mpi_irecv(rbuf1, 4, MPI_REAL8, myrank+nhprocs, 402, MPI_COMM_WORLD, request(3), ierr)

            sbuf(1) = a(n_mpi)
            sbuf(2) = b(n_mpi)
            sbuf(3) = c(n_mpi)
            sbuf(4) = r(n_mpi)
            call mpi_isend(sbuf, 4, MPI_REAL8, myrank+nhprocs, 403, MPI_COMM_WORLD, request(1), ierr)

            call mpi_wait(request(3), status, ierr)
            a(n_mpi+1) = rbuf1(1)
            b(n_mpi+1) = rbuf1(2)
            c(n_mpi+1) = rbuf1(3)
            r(n_mpi+1) = rbuf1(4)

            i = n_mpi
            in = n_mpi+1
            det = b(i)*b(in) - c(i)*a(in)
            x(i) = (r(i)*b(in) - r(in)*c(i))/det
            x(in) = (r(in)*b(i) - r(i)*a(in))/det
            call mpi_wait(request(1), status, ierr)

        else if(myrank==nprocs-1) then
            call mpi_irecv(rbuf0, 4, MPI_REAL8, myrank-nhprocs, 403, MPI_COMM_WORLD, request(4), ierr)

            sbuf(1) = a(n_mpi)
            sbuf(2) = b(n_mpi)
            sbuf(3) = c(n_mpi)
            sbuf(4) = r(n_mpi)
            call mpi_isend(sbuf, 4, MPI_REAL8, myrank-nhprocs, 402, MPI_COMM_WORLD, request(2), ierr)

            call mpi_wait(request(4), status, ierr)
            a(0) = rbuf0(1)
            b(0) = rbuf0(2)
            c(0) = rbuf0(3)
            r(0) = rbuf0(4)

            ip = 0
            i = n_mpi
            det = b(ip)*b(i) - c(ip)*a(i)
            x(ip) = (r(ip)*b(i) - r(i)*c(ip))/det
            x(i) = (r(i)*b(ip) - r(ip)*a(i))/det
            call mpi_wait(request(2), status, ierr)
        endif

    end subroutine cr_forward_single_row

    !> 
    !> @brief   Backward substitution of CR until every MPI process gets solution for its single row.
    !>
    subroutine cr_backward_single_row

        use mpi

        implicit none

        integer(4)  :: i, l
        integer(4)  :: nlevel, nhprocs
        integer(4)  :: ip, in, dist_rank, dist2_rank

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(4)
        integer(4)  :: ierr

        nlevel      = log(dble(nprocs))/log(dble(2))
        nhprocs     = nprocs/2
        dist_rank   = nhprocs/2
        dist2_rank  = nhprocs

        ! Back substitution continues until all ranks obtains a solution on last row.
        do l=nlevel-2, 0, -1

            if(mod((myrank+1),dist2_rank) == 0) then 
                if(myrank+dist_rank<nprocs) then
                    call mpi_isend(x(n_mpi), 1, MPI_REAL8, myrank+dist_rank, 500, MPI_COMM_WORLD, request(1), ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_isend(x(n_mpi), 1, MPI_REAL8, myrank-dist_rank, 501, MPI_COMM_WORLD, request(2), ierr)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_wait(request(1), status, ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_wait(request(2), status, ierr)
                endif
            ! Only Odd rows of each level calculate new solution using a couple of even rows.
            else if(mod((myrank+1),dist2_rank) == dist_rank) then
                if(myrank-dist_rank>=0) then
                    call mpi_irecv(x(0),       1, MPI_REAL8, myrank-dist_rank, 500, MPI_COMM_WORLD, request(3), ierr)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_irecv(x(n_mpi+1), 1, MPI_REAL8, myrank+dist_rank, 501, MPI_COMM_WORLD, request(4), ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_wait(request(3), status, ierr)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_wait(request(4), status, ierr)
                endif

                i=n_mpi
                ip = 0
                in = n_mpi+1
                x(i) = r(i)-c(i)*x(in)-a(i)*x(ip)
                x(i) = x(i)/b(i)
            endif
            dist_rank  = dist_rank / 2
            dist2_rank = dist_rank / 2
        enddo

    end subroutine cr_backward_single_row

    !> 
    !> @brief   PCR between a single row per MPI process and 2x2 matrix solver between i and i+nprocs/2 rows. 
    !>
    subroutine pcr_forward_single_row

        use mpi

        implicit none

        integer(4)  :: i, l
        integer(4)  :: nlevel, nhprocs
        integer(4)  :: myrank_level, nprocs_level
        integer(4)  :: ip, in, dist_rank, dist2_rank
        real(8)     :: alpha, gamma, det
        real(8)     :: sbuf(4), rbuf0(4), rbuf1(4)

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(4)
        integer(4)  :: ierr

        nlevel      = log(dble(nprocs))/log(dble(2))
        nhprocs     = nprocs/2
        dist_rank   = 1
        dist2_rank  = 2

        ! Parallel cyclic reduction continues until 2x2 matrix are made between a pair of rank, 
        ! (myrank, myrank+nhprocs).
        do l = 0, nlevel-2

            ! Rank is newly calculated in each level to find communication pair.
            ! Nprocs is also newly calculated as myrank is changed.
            myrank_level = myrank / dist_rank
            nprocs_level = nprocs / dist_rank

            sbuf(1) = a(n_mpi)
            sbuf(2) = b(n_mpi)
            sbuf(3) = c(n_mpi)
            sbuf(4) = r(n_mpi)

            ! All rows exchange data for reduction and perform reduction successively.
            ! Coefficients are updated for every rows.
            if(mod((myrank_level+1), 2) == 0) then
                if(myrank+dist_rank<nprocs) then
                    call mpi_irecv(rbuf1, 4, MPI_REAL8, myrank+dist_rank, 202, MPI_COMM_WORLD, request(1), ierr)
                    call mpi_isend(sbuf,  4, MPI_REAL8, myrank+dist_rank, 203, MPI_COMM_WORLD, request(2), ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_irecv(rbuf0, 4, MPI_REAL8, myrank-dist_rank, 200, MPI_COMM_WORLD, request(3), ierr)
                    call mpi_isend(sbuf,  4, MPI_REAL8, myrank-dist_rank, 201, MPI_COMM_WORLD, request(4), ierr)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_wait(request(1), status, ierr)
                    a(n_mpi+1) = rbuf1(1)
                    b(n_mpi+1) = rbuf1(2)
                    c(n_mpi+1) = rbuf1(3)
                    r(n_mpi+1) = rbuf1(4)
                    call mpi_wait(request(2), status, ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_wait(request(3), status, ierr)
                    a(0) = rbuf0(1)
                    b(0) = rbuf0(2)
                    c(0) = rbuf0(3)
                    r(0) = rbuf0(4)
                    call mpi_wait(request(4), status, ierr)
                endif
            else if(mod((myrank_level+1), 2) == 1) then
                if(myrank+dist_rank<nprocs) then
                    call mpi_irecv(rbuf1, 4, MPI_REAL8, myrank+dist_rank, 201, MPI_COMM_WORLD, request(1), ierr)
                    call mpi_isend(sbuf,  4, MPI_REAL8, myrank+dist_rank, 200, MPI_COMM_WORLD, request(2), ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_irecv(rbuf0, 4, MPI_REAL8, myrank-dist_rank, 203, MPI_COMM_WORLD, request(3), ierr)
                    call mpi_isend(sbuf,  4, MPI_REAL8, myrank-dist_rank, 202, MPI_COMM_WORLD, request(4), ierr)
                endif
                if(myrank+dist_rank<nprocs) then
                    call mpi_wait(request(1), status, ierr)
                    a(n_mpi+1) = rbuf1(1)
                    b(n_mpi+1) = rbuf1(2)
                    c(n_mpi+1) = rbuf1(3)
                    r(n_mpi+1) = rbuf1(4)
                    call mpi_wait(request(2), status, ierr)
                endif
                if(myrank-dist_rank>=0) then
                    call mpi_wait(request(3), status, ierr)
                    a(0) = rbuf0(1)
                    b(0) = rbuf0(2)
                    c(0) = rbuf0(3)
                    r(0) = rbuf0(4)
                    call mpi_wait(request(4), status, ierr)
                endif
            endif

            i = n_mpi
            ip = 0
            in = i + 1
            if(myrank_level == 0) then
                alpha = 0.0
            else 
                alpha = -a(i) / b(ip)
            endif
            if(myrank_level == nprocs_level-1) then
                gamma = 0.0
            else
                gamma = -c(i) / b(in)
            endif

            b(i) = b(i) + alpha * c(ip) + gamma * a(in)
            a(i) = alpha * a(ip)
            c(i) = gamma * c(in)
            r(i) = r(i) + alpha * r(ip) + gamma * r(in)

            dist_rank  = dist_rank * 2
            dist2_rank = dist2_rank * 2
        enddo

        ! Solving 2x2 matrix. All pair of ranks, myrank and myrank+nhprocs, solves it simultaneously.

        sbuf(1) = a(n_mpi)
        sbuf(2) = b(n_mpi)
        sbuf(3) = c(n_mpi)
        sbuf(4) = r(n_mpi)
        if(myrank<nhprocs) then
            call mpi_irecv(rbuf1, 4, MPI_REAL8, myrank+nhprocs, 300, MPI_COMM_WORLD, request(1), ierr)
            call mpi_isend(sbuf,  4, MPI_REAL8, myrank+nhprocs, 301, MPI_COMM_WORLD, request(2), ierr)

            call mpi_wait(request(1), status, ierr)
            a(n_mpi+1) = rbuf1(1)
            b(n_mpi+1) = rbuf1(2)
            c(n_mpi+1) = rbuf1(3)
            r(n_mpi+1) = rbuf1(4)

            i = n_mpi
            in = n_mpi+1

            det = b(i)*b(in) - c(i)*a(in)
            x(i) = (r(i)*b(in) - r(in)*c(i))/det
            x(in) = (r(in)*b(i) - r(i)*a(in))/det
            call mpi_wait(request(2), status, ierr)
        else if(myrank>=nhprocs) then
            call mpi_irecv(rbuf0, 4, MPI_REAL8, myrank-nhprocs, 301, MPI_COMM_WORLD, request(3), ierr)
            call mpi_isend(sbuf,  4, MPI_REAL8, myrank-nhprocs, 300, MPI_COMM_WORLD, request(4), ierr)

            call mpi_wait(request(3), status, ierr)
            a(0) = rbuf0(1)
            b(0) = rbuf0(2)
            c(0) = rbuf0(3)
            r(0) = rbuf0(4)

            ip = 0
            i = n_mpi

            det = b(ip)*b(i) - c(ip)*a(i)
            x(ip) = (r(ip)*b(i) - r(i)*c(ip))/det
            x(i) = (r(i)*b(ip) - r(ip)*a(i))/det
            call mpi_wait(request(4), status, ierr)
        endif

    end subroutine pcr_forward_single_row


    !> 
    !> @brief   First phase of hybrid Thomas and PCR algorithm
    !> @detail  Forward and backward elimination to remain two equations of first and 
    !>          last rows for each MPI processes 
    subroutine pThomas_forward_multiple_row

        implicit none

        integer(4)  :: i
        real(8)     :: alpha, beta
   
        do i = 3, n_mpi
            alpha = - a(i) / b(i-1)
            a(i) = alpha * a(i-1)
            b(i) = b(i) + alpha * c(i-1)
            r(i) = r(i) + alpha * r(i-1)
        enddo

        do i = n_mpi-2, 1, -1
            beta = - c(i) / b(i+1)
            c(i) = beta * c(i+1)
            r(i) = r(i) + beta * r(i+1)
            if(i==1) then
                b(1) = b(1) + beta * a(2)
            else
                a(i) = a(i) + beta * a(i+1)
            endif
        enddo

    end subroutine pThomas_forward_multiple_row
   
    !> 
    !> @brief   PCR solver for two equations per each MPI process
    !> @detail  Forward CR to remain a single equation per each MPI process.
    !>          PCR solver for single row is, then, executed.
    !>          Substitution is also performed to obtain every solution.
    !>
    subroutine pcr_double_row_substitution

        use mpi

        implicit none

        integer(4)  :: i, ip, in
        real(8)     :: alpha, gamma
        real(8)     :: sbuf(4), rbuf(4)

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(2)
        integer(4)  :: ierr

        !< Cyclic reduction until single row remains per MPI process.
        !< First row of next rank is sent to current rank at the row of n_mpi+1 for reduction.
        if(myrank<nprocs-1) then
            call mpi_irecv(rbuf, 4, MPI_REAL8, myrank+1, 0, MPI_COMM_WORLD, request(1), ierr)
        endif
        if(myrank>0) then
            sbuf(1) = a(1);
            sbuf(2) = b(1);
            sbuf(3) = c(1);
            sbuf(4) = r(1);
            call mpi_isend(sbuf, 4, MPI_REAL8, myrank-1, 0, MPI_COMM_WORLD, request(2), ierr)
        endif
        if(myrank<nprocs-1) then
            call mpi_wait(request(1), status, ierr)
            a(n_mpi+1) = rbuf(1)
            b(n_mpi+1) = rbuf(2)
            c(n_mpi+1) = rbuf(3)
            r(n_mpi+1) = rbuf(4)
        endif
   
        !< Every first row are reduced to the last row (n_mpi) in each MPI rank.
        i = n_mpi
        ip = 1
        in = i + 1
        alpha = -a(i) / b(ip)
        gamma = -c(i) / b(in)

        b(i) = b(i) + (alpha * c(ip) + gamma * a(in))
        a(i) = alpha * a(ip)
        c(i) = gamma * c(in)
        r(i) = r(i) + (alpha * r(ip) + gamma * r(in))
        
        if(myrank>0) then
            call mpi_wait(request(2), status, ierr)
        endif
   
        !< Solution of last row in each MPI rank is obtained in pcr_forward_single_row().
        call pcr_forward_single_row

        !< Solution of first row in each MPI rank.
        if(myrank>0) then
            call mpi_irecv(x(0),   1, MPI_REAL8, myrank-1, 100, MPI_COMM_WORLD, request(1), ierr)
        endif
        if(myrank<nprocs-1) then
            call mpi_isend(x(n_mpi), 1, MPI_REAL8, myrank+1, 100, MPI_COMM_WORLD, request(2), ierr)
        endif
        if(myrank>0) then
            call mpi_wait(request(1), status, ierr)
        endif
        i = 1
        ip = 0
        in = n_mpi
        x(1) = r(1)-c(1)*x(n_mpi)-a(1)*x(0)
        x(1) = x(1)/b(1)

        if(myrank<nprocs-1) then
            call mpi_wait(request(2), status, ierr)
        endif
        !< Solution of other rows in each MPI rank.
        do i = 2, n_mpi-1
            x(i) = r(i)-c(i)*x(n_mpi)-a(i)*x(1)
            x(i) = x(i)/b(i)
        enddo
    end subroutine pcr_double_row_substitution
   

    !> 
    !> @brief   Solution check
    !> @param   *a_ver Coefficients of a with original values
    !> @param   *b_ver Coefficients of b with original values
    !> @param   *c_ver Coefficients of c with original values
    !> @param   *r_ver RHS vector with original values
    !> @param   *x_sol Solution vector
    !>
    subroutine verify_solution(a_ver, b_ver, c_ver, r_ver, x_sol)

        use mpi

        implicit none

        real(8), intent(inout), target :: a_ver(0:n_mpi+1)
        real(8), intent(inout), target :: b_ver(0:n_mpi+1)
        real(8), intent(inout), target :: c_ver(0:n_mpi+1)
        real(8), intent(inout), target :: r_ver(0:n_mpi+1)
        real(8), intent(inout), target :: x_sol(0:n_mpi+1)

        integer(4)  :: i
        real(8), allocatable, dimension(:)     :: y

        integer(4)  :: status(MPI_STATUS_SIZE)
        integer(4)  :: request(4)
        integer(4)  :: ierr

        a => a_ver
        b => b_ver
        c => c_ver
        r => r_ver
        x => x_sol

        allocate(y(n_mpi+2))

        if(myrank>0) then
            call mpi_isend(x(1), 1, MPI_REAL8, myrank-1, 900, MPI_COMM_WORLD, request(1), ierr)
            call mpi_irecv(x(0), 1, MPI_REAL8, myrank-1, 901, MPI_COMM_WORLD, request(2), ierr)
        endif
        if(myrank<nprocs-1) then
            call mpi_isend(x(n_mpi),   1, MPI_REAL8, myrank+1, 901, MPI_COMM_WORLD, request(3), ierr)
            call mpi_irecv(x(n_mpi+1), 1, MPI_REAL8, myrank+1, 900, MPI_COMM_WORLD, request(4), ierr)
        endif

        if(myrank>0) then
            call mpi_wait(request(1), status, ierr)
            call mpi_wait(request(2), status, ierr)
        endif
        if(myrank<nprocs-1) then
            call mpi_wait(request(3), status, ierr)
            call mpi_wait(request(4), status, ierr)
        endif
        
        do i = 1, n_mpi
            y(i) = a(i)*x(i-1)+b(i)*x(i)+c(i)*x(i+1)
!            print '("Verify solution1 : myrank = ", i3, " a=", f12.6, " b=", f12.6, " c=", f12.6, " x=", f12.6, " r[", i3,"]=", f12.6, " y[", i3,"]=", f12.6)', myrank,a(i),b(i),c(i),x(i),i+n_mpi*myrank,r(i),i+n_mpi*myrank,y(i)
            print '("Verify solution1 : myrank = ", i3, " x=", f15.9, " r[", i3,"]=", f15.9, " y[", i3,"]=", f15.9)', myrank,x(i),i+n_mpi*myrank,r(i),i+n_mpi*myrank,y(i)
        enddo
        deallocate(y)

        nullify(a, b, c, r, x)

    end subroutine verify_solution

end module tdma_parallel
