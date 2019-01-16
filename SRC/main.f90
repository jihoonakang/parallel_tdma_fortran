!>
!> @brief       Parallel TDMA test subroutine
!> @author      Ji-Hoon Kang (jhkang@kisti.re.kr), Korea Institute of Science and Technology Information
!> @date        20 January 2019
!> @version     0.1
!> @par         Copyright
!>              Copyright (c) 2018 by Ji-Hoon Kang. All rights reserved.
!> @par         License     
!>              This project is release under the terms of the MIT License (see LICENSE in )
!>
program main

    use mpi
    use tdma_parallel

    implicit none

    integer(kind=4), parameter :: n = 32
    integer(kind=4)     :: i
    integer(kind=4)     :: nprocs, myrank
    integer(kind=4)     :: n_mpi
    integer(kind=4)     :: ierr

    real(kind=8), allocatable, dimension(:) :: a_mpi, b_mpi, c_mpi, x_mpi, r_mpi
    real(kind=8), allocatable, dimension(:) :: a_ver, b_ver, c_ver, r_ver

    call mpi_init(ierr)
    call mpi_comm_size(MPI_COMM_WORLD, nprocs, ierr)
    call mpi_comm_rank(MPI_COMM_WORLD, myrank, ierr)

    n_mpi = n / nprocs

    allocate(a_mpi(0:n_mpi+1))
    allocate(b_mpi(0:n_mpi+1))
    allocate(c_mpi(0:n_mpi+1))
    allocate(r_mpi(0:n_mpi+1))
    allocate(x_mpi(0:n_mpi+1))

    allocate(a_ver(0:n_mpi+1))
    allocate(b_ver(0:n_mpi+1))
    allocate(c_ver(0:n_mpi+1))
    allocate(r_ver(0:n_mpi+1))

    do i = 0, n_mpi+1
        a_mpi(i) = 0.0
        b_mpi(i) = 1.0
        c_mpi(i) = 0.0
        r_mpi(i) = 0.0
        x_mpi(i) = 0.0
    enddo

    do i = 1, n_mpi
        a_mpi(i) = 1.0 !+0.01*(i+myrank*n_mpi)
        c_mpi(i) = 1.0 !+0.02*(i+myrank*n_mpi)
        b_mpi(i) = -(a_mpi(i)+c_mpi(i))-0.1 !-0.02*(i+myrank*n_mpi)*(i+myrank*n_mpi)
        r_mpi(i) = dble(i-1+myrank*n_mpi)
        a_ver(i) = a_mpi(i)
        b_ver(i) = b_mpi(i)
        c_ver(i) = c_mpi(i)
        r_ver(i) = r_mpi(i)
    enddo

    call tdma_setup    (n, nprocs, myrank)
    call Thomas_pcr_solver (a_mpi, b_mpi, c_mpi, r_mpi, x_mpi)
!    call cr_pcr_solver (a_mpi, b_mpi, c_mpi, r_mpi, x_mpi)
    call verify_solution(a_ver, b_ver, c_ver, r_ver, x_mpi)

    deallocate(a_mpi)
    deallocate(b_mpi)
    deallocate(c_mpi)
    deallocate(r_mpi)
    deallocate(x_mpi)
    deallocate(a_ver)
    deallocate(b_ver)
    deallocate(c_ver)
    deallocate(r_ver)

    call mpi_finalize(ierr)

end program