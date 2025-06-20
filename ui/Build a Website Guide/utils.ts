import { cn } from "@/lib/utils"

export function Utils() {
  return {
    cn,
  }
}

/**
 * Utility function to merge class names with tailwind-merge
 */
export function cn(...inputs: any[]): string {
  // Simple implementation of class name merging
  return inputs.filter(Boolean).join(" ")
}
