type Props = {
  size?: number;
  className?: string;
};

/**
 * Sinew brand mark — three offset rounded bars, like woven fibers.
 * A nod to the meaning of "sinew" (tendon / fiber).
 */
export function SinewMark({ size = 22, className }: Props) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      aria-hidden="true"
    >
      <rect x="4" y="5.5"  width="13" height="2.6" rx="1.3" fill="currentColor" />
      <rect x="7" y="10.7" width="13" height="2.6" rx="1.3" fill="currentColor" />
      <rect x="4" y="15.9" width="13" height="2.6" rx="1.3" fill="currentColor" />
    </svg>
  );
}
