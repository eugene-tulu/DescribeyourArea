import { useState } from "react";
import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function CopySummary({ summaryText }: { summaryText: string }) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async () => {
    if (summaryText) {
      try {
        await navigator.clipboard.writeText(summaryText);
        setCopied(true);

        // reset after 2s
        setTimeout(() => setCopied(false), 2000);
      } catch (error) {
        console.error("Copy error:", error);
      }
    }
  };

  return (
    <Button
      onClick={copyToClipboard}
      variant="outline"
      size="sm"
      className="border-white/30 text-white hover:bg-white/10"
    >
      <Copy className="w-4 h-4 mr-1" />
      {copied ? "Copied!" : "Copy"}
    </Button>
  );
}
