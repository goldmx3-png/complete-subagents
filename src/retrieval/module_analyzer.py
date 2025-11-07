"""
Module analysis utilities for understanding hierarchical distribution in retrieved chunks.

This module helps detect cross-module queries and provides insights into module distribution
to enable better context organization and LLM instruction adaptation.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def analyze_module_distribution(chunks: List[Dict]) -> Dict:
    """
    Analyze the distribution of chunks across modules and subsections.

    Args:
        chunks: List of retrieved chunks with hierarchy metadata

    Returns:
        Dictionary with comprehensive module distribution analysis:
        {
            'total_chunks': int,
            'num_modules': int,
            'num_subsections': int,
            'modules': {module_name: count},
            'subsections': {subsection_name: count},
            'is_cross_module': bool,
            'primary_module': str,
            'diversity_score': float (0-1, higher = more diverse),
            'depth_stats': {'min': int, 'max': int, 'avg': float},
            'breadcrumb_samples': [str]
        }
    """
    if not chunks:
        return {
            'total_chunks': 0,
            'num_modules': 0,
            'num_subsections': 0,
            'modules': {},
            'subsections': {},
            'is_cross_module': False,
            'primary_module': None,
            'diversity_score': 0.0,
            'depth_stats': {'min': 0, 'max': 0, 'avg': 0.0},
            'breadcrumb_samples': []
        }

    module_counter = Counter()
    subsection_counter = Counter()
    depths = []
    breadcrumb_samples = []

    for chunk in chunks:
        payload = chunk.get('payload', {})
        metadata = payload.get('metadata', {})
        hierarchy = metadata.get('hierarchy', {})

        if hierarchy:
            # Count modules (h1/root_section)
            root_section = hierarchy.get('root_section', '')
            if root_section and root_section != 'unknown':
                module_counter[root_section] += 1

            # Count subsections (h2/parent_section)
            parent_section = hierarchy.get('parent_section', '')
            if parent_section and parent_section != root_section:
                subsection_counter[parent_section] += 1

            # Collect depth information
            depth = hierarchy.get('depth', 0)
            if depth > 0:
                depths.append(depth)

            # Sample breadcrumbs (up to 5 unique)
            full_path = hierarchy.get('full_path', '')
            if full_path and full_path not in breadcrumb_samples and len(breadcrumb_samples) < 5:
                breadcrumb_samples.append(full_path)

    # Calculate statistics
    num_modules = len(module_counter)
    is_cross_module = num_modules > 1

    # Primary module (most chunks)
    primary_module = module_counter.most_common(1)[0][0] if module_counter else None

    # Calculate diversity score (Shannon entropy normalized to 0-1)
    diversity_score = _calculate_diversity_score(module_counter, len(chunks))

    # Depth statistics
    depth_stats = {
        'min': min(depths) if depths else 0,
        'max': max(depths) if depths else 0,
        'avg': sum(depths) / len(depths) if depths else 0.0
    }

    analysis = {
        'total_chunks': len(chunks),
        'num_modules': num_modules,
        'num_subsections': len(subsection_counter),
        'modules': dict(module_counter),
        'subsections': dict(subsection_counter),
        'is_cross_module': is_cross_module,
        'primary_module': primary_module,
        'diversity_score': diversity_score,
        'depth_stats': depth_stats,
        'breadcrumb_samples': breadcrumb_samples
    }

    logger.info(
        f"Module analysis: {num_modules} modules, "
        f"cross-module={is_cross_module}, "
        f"diversity={diversity_score:.2f}, "
        f"primary={primary_module}"
    )

    return analysis


def detect_cross_module_query(chunks: List[Dict], threshold: float = 0.3) -> Tuple[bool, Optional[str]]:
    """
    Detect if a query spans multiple modules with significant representation.

    Args:
        chunks: Retrieved chunks
        threshold: Minimum proportion for secondary module to be considered significant (0-1)

    Returns:
        Tuple of (is_cross_module: bool, reason: str or None)
    """
    analysis = analyze_module_distribution(chunks)

    if not analysis['is_cross_module']:
        return False, None

    modules = analysis['modules']
    total = analysis['total_chunks']

    # Check if secondary modules have significant representation
    sorted_modules = sorted(modules.items(), key=lambda x: x[1], reverse=True)

    if len(sorted_modules) < 2:
        return False, None

    primary_name, primary_count = sorted_modules[0]
    secondary_name, secondary_count = sorted_modules[1]

    secondary_proportion = secondary_count / total

    if secondary_proportion >= threshold:
        reason = (
            f"Cross-module query detected: "
            f"{primary_name} ({primary_count} chunks), "
            f"{secondary_name} ({secondary_count} chunks, {secondary_proportion:.1%})"
        )
        return True, reason

    return False, f"Primary module dominates: {primary_name} ({primary_count}/{total})"


def get_module_groups(chunks: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group chunks by their primary module (h1/root_section).

    Args:
        chunks: Retrieved chunks

    Returns:
        Dictionary mapping module names to lists of chunks
    """
    module_groups = defaultdict(list)

    for chunk in chunks:
        payload = chunk.get('payload', {})
        metadata = payload.get('metadata', {})
        hierarchy = metadata.get('hierarchy', {})

        if hierarchy:
            module = hierarchy.get('root_section', 'Unknown')
        else:
            # Fallback to old metadata
            module = payload.get('section_title', 'Unknown')

        module_groups[module].append(chunk)

    return dict(module_groups)


def balance_module_representation(
    chunks: List[Dict],
    max_per_module: int = 3,
    preserve_top_k: int = 2
) -> List[Dict]:
    """
    Balance chunk distribution across modules to ensure fair representation.

    Args:
        chunks: Retrieved chunks (assumed sorted by relevance)
        max_per_module: Maximum chunks to keep per module
        preserve_top_k: Always keep top K most relevant chunks regardless of module

    Returns:
        Rebalanced list of chunks
    """
    if not chunks:
        return []

    # Always preserve top K most relevant
    preserved = chunks[:preserve_top_k]
    remaining = chunks[preserve_top_k:]

    # Group remaining chunks by module
    module_groups = defaultdict(list)
    for chunk in remaining:
        payload = chunk.get('payload', {})
        metadata = payload.get('metadata', {})
        hierarchy = metadata.get('hierarchy', {})

        module = hierarchy.get('root_section', 'Unknown') if hierarchy else 'Unknown'
        module_groups[module].append(chunk)

    # Take up to max_per_module from each module
    balanced = list(preserved)
    for module, module_chunks in sorted(module_groups.items()):
        balanced.extend(module_chunks[:max_per_module])

    logger.info(f"Balanced {len(chunks)} chunks to {len(balanced)} with max {max_per_module} per module")

    return balanced


def _calculate_diversity_score(module_counter: Counter, total_chunks: int) -> float:
    """
    Calculate diversity score using Shannon entropy.

    Args:
        module_counter: Counter of chunks per module
        total_chunks: Total number of chunks

    Returns:
        Diversity score normalized to 0-1 (0 = single module, 1 = perfectly distributed)
    """
    if not module_counter or len(module_counter) <= 1:
        return 0.0

    import math

    entropy = 0.0
    for count in module_counter.values():
        if count > 0:
            proportion = count / total_chunks
            entropy -= proportion * math.log2(proportion)

    # Normalize to 0-1 (max entropy for N modules is log2(N))
    max_entropy = math.log2(len(module_counter))
    normalized_score = entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_score


def format_module_summary(analysis: Dict) -> str:
    """
    Format module analysis into a human-readable summary.

    Args:
        analysis: Result from analyze_module_distribution()

    Returns:
        Formatted string summary
    """
    lines = [
        f"Total chunks: {analysis['total_chunks']}",
        f"Modules: {analysis['num_modules']}",
        f"Cross-module: {'Yes' if analysis['is_cross_module'] else 'No'}",
        f"Diversity score: {analysis['diversity_score']:.2f}"
    ]

    if analysis['modules']:
        lines.append("\nModule distribution:")
        for module, count in sorted(analysis['modules'].items(),
                                    key=lambda x: x[1],
                                    reverse=True):
            proportion = count / analysis['total_chunks']
            lines.append(f"  - {module}: {count} chunks ({proportion:.1%})")

    if analysis['breadcrumb_samples']:
        lines.append("\nSample paths:")
        for path in analysis['breadcrumb_samples']:
            lines.append(f"  - {path}")

    return "\n".join(lines)
