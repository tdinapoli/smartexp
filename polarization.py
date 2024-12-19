from typing import Literal

import numpy as np

type MuellerMatrix = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.float64]]
type StokesVector = np.ndarray[tuple[Literal[4], ], np.dtype[np.float64]]

def rotation_matrix(phi: float) -> MuellerMatrix:
    """
    Generate the rotation matrix for a given angle phi (in degrees).

    Parameters
    ----------
    phi : float
        Rotation angle in radias.

    Returns
    -------
    numpy.ndarray
        A 4x4 rotation matrix.
    """
    phi_rad = np.radians(phi)
    cos_2phi = np.cos(2 * phi)
    sin_2phi = np.sin(2 * phi)

    return np.array([
        [1, 0, 0, 0],
        [0, cos_2phi, sin_2phi, 0],
        [0, -sin_2phi, cos_2phi, 0],
        [0, 0, 0, 1]
    ])


def depolarization_matrix(d_coeffs: list[float]) -> MuellerMatrix:
    """
    Generate the depolarization matrix.

    Parameters
    ----------
    d_coeffs : list of float
        List of 9 depolarization coefficients.

    Returns
    -------
    numpy.ndarray
        A 4x4 depolarization matrix.
    """
    frobenius_norm = np.sqrt(sum(c**2 for c in d_coeffs))
    assert frobenius_norm <= 1, "Depolarization coefficients must satisfy Frobenius norm <= 1."

    d11, d12, d13, d21, d22, d23, d31, d32, d33 = d_coeffs
    return np.array([
        [1, 0, 0, 0],
        [0, d11, d12, d13],
        [0, d21, d22, d23],
        [0, d31, d32, d33]
    ])


def generate_valid_depolarization_coeffs() -> list[float]:
    """
    Generate a random set of valid depolarization coefficients satisfying the Frobenius norm constraint.

    Returns
    -------
    list of float
        List of 9 depolarization coefficients.
    """
    coeffs = np.random.uniform(0, 1, 9)
    frobenius_norm = np.sqrt(np.sum(coeffs**2))
    if frobenius_norm > 1:
        coeffs /= frobenius_norm  # Normalize to satisfy Frobenius norm
    return coeffs.tolist()


def dichroism_matrix(ax: float, ay: float, ac: float, a: float) -> MuellerMatrix:
    """
    Generate the dichroism matrix.

    Parameters
    ----------
    ax : float
        Absorption coefficient for horizontal polarization.
    ay : float
        Absorption coefficient for vertical polarization.
    ac : float
        Absorption coefficient for circular polarization.
    a : float
        Isotropic absorption coefficient.

    Returns
    -------
    numpy.ndarray
        A 4x4 dichroism matrix.
    """
    return np.array([
        [np.exp(-a), np.exp(-ax), 0, np.exp(-ac)],
        [np.exp(-ax), np.exp(-ax), 0, 0],
        [0, 0, np.exp(-ay), 0],
        [np.exp(-ac), 0, 0, np.exp(-ac)]
    ])


def birefringence_matrix(delta: float, theta: float) -> MuellerMatrix:
    """
    Generate the birefringence matrix.

    Parameters
    ----------
    delta : float
        Linear birefringence phase shift (in radians).
    theta : float
        Circular birefringence phase shift (in radians).

    Returns
    -------
    numpy.ndarray
        A 4x4 birefringence matrix.
    """
    M_linear = np.array([
        [1, 0, 0, 0],
        [0, np.cos(delta), np.sin(delta), 0],
        [0, -np.sin(delta), np.cos(delta), 0],
        [0, 0, 0, 1]
    ])

    M_circular = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), 0, np.sin(theta)],
        [0, 0, 1, 0],
        [0, -np.sin(theta), 0, np.cos(theta)]
    ])

    return M_linear @ M_circular


def combined_mueller_matrix(phi: float, d_coeffs: list[float], ax: float, ay: float, ac: float, a: float, delta: float, theta: float) -> MuellerMatrix:
    """
    Generate the full Mueller matrix including rotation, depolarization, dichroism, and birefringence.

    Parameters
    ----------
    phi : float
        Material orientation angle in degrees.
    d_coeffs : list of float
        List of 9 depolarization coefficients.
    ax : float
        Absorption coefficient for horizontal polarization.
    ay : float
        Absorption coefficient for vertical polarization.
    ac : float
        Absorption coefficient for circular polarization.
    a : float
        Isotropic absorption coefficient.
    delta : float
        Linear birefringence phase shift (in radians).
    theta : float
        Circular birefringence phase shift (in radians).

    Returns
    -------
    numpy.ndarray
        The full 4x4 Mueller matrix.
    """
    R_neg_phi = rotation_matrix(-phi)
    R_phi = rotation_matrix(phi)
    M_depol = depolarization_matrix(d_coeffs)
    M_dichroism = dichroism_matrix(ax, ay, ac, a)
    M_birefringence = birefringence_matrix(delta, theta)
    return R_neg_phi @ M_depol @ M_dichroism @ M_birefringence @ R_phi


def simulate_measurements(mueller_matrix: MuellerMatrix, measurement_config: list[tuple[StokesVector, float]])  -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    """
    Simulate output intensities for a given Mueller matrix and measurement configuration.

    Parameters
    ----------
    mueller_matrix : numpy.ndarray
        A 4x4 Mueller matrix representing the material.
    measurement_config : list of tuples
        Each tuple contains:
        - input_stokes (numpy.ndarray): A 1D array representing the input Stokes vector.
        - analyzer_orientation (float): The analyzer orientation in degrees.

    Returns
    -------
    numpy.ndarray
        Simulated output intensities for the provided measurement configuration.
    """
    output_intensities = []
    for input_stokes, analyzer_orientation in measurement_config:
        analyzer_stokes = np.array([
            1,
            np.cos(2 * np.radians(analyzer_orientation)),
            np.sin(2 * np.radians(analyzer_orientation)),
            0
        ])
        stokes_out = mueller_matrix @ input_stokes
        intensity = analyzer_stokes @ stokes_out
        output_intensities.append(intensity)
    return np.array(output_intensities)


def reconstruct_mueller_matrix(measurements: np.ndarray[tuple[int], np.dtype[np.float64]], measurement_config: list[tuple[StokesVector, float]]) -> MuellerMatrix:
    """
    Reconstruct the Mueller matrix from simulated measurements.

    Parameters
    ----------
    measurements : numpy.ndarray
        Array of simulated output intensities.
    measurement_config : list of tuples
        Each tuple contains:
        - input_stokes (numpy.ndarray): The input Stokes vector.
        - analyzer_orientation (float): Analyzer orientation in degrees.

    Returns
    -------
    numpy.ndarray
        The reconstructed 4x4 Mueller matrix.
    """
    A = []
    b = []

    for i, (input_stokes, analyzer_orientation) in enumerate(measurement_config):
        analyzer_stokes = np.array([
            1,
            np.cos(2 * np.radians(analyzer_orientation)),
            np.sin(2 * np.radians(analyzer_orientation)),
            0
        ])
        row = np.kron(analyzer_stokes, input_stokes)
        A.append(row)
        b.append(measurements[i])

    A = np.array(A)
    b = np.array(b)
    m = np.linalg.lstsq(A, b, rcond=None)[0]
    return m.reshape(4, 4)


def compare_matrices(true_matrix: np.ndarray, reconstructed_matrix: np.ndarray):
    """
    Compare the true Mueller matrix with the reconstructed Mueller matrix.

    Parameters
    ----------
    true_matrix : numpy.ndarray
        The original Mueller matrix used for simulation.
    reconstructed_matrix : numpy.ndarray
        The Mueller matrix reconstructed from simulated measurements.

    Returns
    -------
    dict
        A dictionary containing:
        - "difference": The absolute difference between the matrices.
        - "error_norm": The Frobenius norm of the difference.
    """
    difference = np.abs(true_matrix - reconstructed_matrix)
    error_norm = np.linalg.norm(difference, ord='fro')

    return {
        "difference": difference,
        "error_norm": error_norm
    }


def extract_parameters(mueller_matrix: MuellerMatrix) -> dict[str, float]:
    """
    Extract physical parameters (phi, d_coeffs, ax, ay, ac, a, delta, theta) from the Mueller matrix.

    Parameters
    ----------
    mueller_matrix : numpy.ndarray
        The reconstructed 4x4 Mueller matrix.

    Returns
    -------
    dict
        Extracted parameters.
    """
    phi = 0.5 * np.arctan2(mueller_matrix[1, 2], mueller_matrix[1, 1])
    phi_deg = np.degrees(phi)

    R_neg_phi = rotation_matrix(-phi_deg)
    M_no_rotation = R_neg_phi @ mueller_matrix @ rotation_matrix(phi_deg)

    depol_submatrix = M_no_rotation[1:, 1:]
    frobenius_norm = np.sqrt(np.sum(depol_submatrix**2))
    assert frobenius_norm <= 1, "Depolarization coefficients violate Frobenius norm constraint."
    d_coeffs = depol_submatrix.flatten()

    ax = -np.log(M_no_rotation[0, 1])
    ay = -np.log(M_no_rotation[0, 2])
    ac = -np.log(M_no_rotation[0, 3])
    a = -np.log(M_no_rotation[0, 0])

    birefringence_submatrix = M_no_rotation[1:, 1:]
    delta = np.arctan2(birefringence_submatrix[1, 2], birefringence_submatrix[1, 1])
    theta = np.arctan2(birefringence_submatrix[0, 2], birefringence_submatrix[0, 0])

    return {
        "phi": phi_deg,
        "d_coeffs": d_coeffs.tolist(),
        "ax": ax,
        "ay": ay,
        "ac": ac,
        "a": a,
        "delta": delta,
        "theta": theta
    }

def main():
    """
    Main function to demonstrate the Mueller matrix generation, simulation, reconstruction, and comparison.
    """
    phi = 30
    d_coeffs = generate_valid_depolarization_coeffs()
    ax, ay, ac, a = 0.1, 0.2, 0.05, 0.15
    delta = np.pi / 4
    theta = np.pi / 6

    M = combined_mueller_matrix(phi, d_coeffs, ax, ay, ac, a, delta, theta)
    print(extract_parameters(M))

    input_stokes: list[StokesVector] = [
        np.array([1, 1, 0, 0]),
        np.array([1, -1, 0, 0]),
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 0, 1])
    ]
    analyzer_orientations = [0, 45, 90, 135]
    measurement_config = [(s, a) for s in input_stokes for a in analyzer_orientations]

    measurements = simulate_measurements(M, measurement_config)

    M_reconstructed = reconstruct_mueller_matrix(measurements, measurement_config)

    print("Original Mueller Matrix:")
    print(M)
    print("\nReconstructed Mueller Matrix:")
    print(M_reconstructed)

    comparison = compare_matrices(M, M_reconstructed)
    print("\nComparison Results:")
    print(f"Difference:\n{comparison['difference']}")
    print(f"Error Frobenius Norm: {comparison['error_norm']}")

if __name__ == "__main__":
    main()