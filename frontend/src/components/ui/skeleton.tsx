import { cn } from "@/lib/utils";
// NOTE  MC8yOmFIVnBZMlhvaklQb3RvVTZjRFZVT1E9PToxMTZkYzFmOQ==

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  );
}
// TODO  MS8yOmFIVnBZMlhvaklQb3RvVTZjRFZVT1E9PToxMTZkYzFmOQ==

export { Skeleton };
